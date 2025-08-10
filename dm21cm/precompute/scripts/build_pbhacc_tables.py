"""Build PBH accretion table."""

import os
import sys
import argparse

import numpy as np
import astropy.units as u
import astropy.constants as c
from astropy.cosmology import Planck18 as cosmo
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from tqdm import tqdm
import logging

WDIR = os.environ['DM21CM_DIR']
sys.path.append(WDIR)
from dm21cm.utils import save_h5_dict, load_h5_dict
from dm21cm.precompute.accretion import PBHAccretionModel
from dm21cm.precompute.halo import cmz
from dm21cm.precompute.ps import RHO_M


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--log10mPBH', type=float)
    parser.add_argument('--save_memory', action='store_true', default=False)
    args = parser.parse_args()

    #===== Initialization =====
    hmfdata = load_h5_dict("/n/holystore01/LABS/iaifi_lab/Users/yitians/dm21cm/data/hmf/hmf.h5")
    z_s = hmfdata['z'] # [1]    | redshift
    d_s = hmfdata['d'] # [1]    | delta (overdensity)
    m_s = hmfdata['m'] # [Msun] | halo mass

    dsub_s = np.linspace(-1, 1.6, 64) # [1] | delta (overdensity)
    zfull_s = np.concatenate((z_s, np.geomspace(z_s[-1], 3000, 300)[1:])) # for Unconditional PS and ST tables
    cinf_s = np.geomspace(
        np.sqrt(5/3 * (1+0.) * 10  * u.K * c.k_B / c.m_p).to(u.km/u.s).value, # 0.371 km/s
        np.sqrt(5/3 * (1+1.) * 1e4 * u.K * c.k_B / c.m_p).to(u.km/u.s).value, # 16.6 km/s
        64
    ) # should cover redshift 4-3000
    vcb_s = np.geomspace(0.01, 100, 128) # [km/s]
    mPBH = 10**args.log10mPBH # [Msun]

    model_kwargs_dict = {
        'PRc23'   : dict(accretion_type='PR-ADAF'),
        'PRc14'   : dict(accretion_type='PR-ADAF', c_in=14),
        'PRc29'   : dict(accretion_type='PR-ADAF', c_in=29),
        'PRc23B'  : dict(accretion_type='PR-ADAF', v_rel_type='DMDM'),
        'PRc23H'  : dict(accretion_type='PRHALO-ADAF'),
        'PRc23dm' : dict(accretion_type='PR-ADAF', delta_e=1e-2),
        'PRc23dp' : dict(accretion_type='PR-ADAF', delta_e=0.5),
        'BHLl2'   : dict(accretion_type='BHL-ADAF', lambda_fudge=1e-2),
    }
    am = PBHAccretionModel(**model_kwargs_dict[args.model])

    # HMF threshold
    m_thres = 30 * mPBH

    #===== File names =====
    run_name = args.model
    run_subname = f'log10m{np.log10(mPBH):.3f}'
    cache_file = f"{WDIR}/data/pbh-accretion/L_table_cache/{run_name}/{run_name}_{run_subname}.npy"
    halo_file  = f"{os.environ['DM21CM_DATA_DIR']}/pbhacc_rates/{run_name}/{run_name}_{run_subname}_halo.h5"
    cosmo_file = f"{os.environ['DM21CM_DATA_DIR']}/pbhacc_rates/{run_name}/{run_name}_{run_subname}_cosmo.h5"
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    os.makedirs(os.path.dirname(halo_file), exist_ok=True)
    os.makedirs(os.path.dirname(cosmo_file), exist_ok=True)

    #===== Halo luminosity table =====
    # L_table: (z, m) [Msun / yr]
    if os.path.exists(cache_file):
        L_table = np.load(cache_file)
        print(f"Using cached {cache_file}.")
    else:
        print(f"Cache not found. Creating {cache_file}...")
        L_table = np.zeros((len(z_s), len(m_s)))
            
        L_halo_vmap = jax.jit(jax.vmap(am.L_halo, in_axes=(None, 0, 0, None)))
        L_halo_jit = jax.jit(am.L_halo)
        for i_z, z in enumerate(tqdm(z_s)):
            m_halo_s = jnp.asarray(m_s) # [Msun]
            c_halo_s = jnp.asarray(cmz(m_s, z))
            if args.save_memory:
                for i_m, m in enumerate(m_s):
                    L_table[i_z, i_m] = L_halo_jit(mPBH, m, c_halo_s[i_m], z)
            else:
                L_table[i_z] = L_halo_vmap(mPBH, m_halo_s, c_halo_s, z)
        np.save(cache_file, L_table)


    #===== Halo PBH: Summing HMF =====
    # xxx_table: [Msun/yr / cMpc^3]
    print("Halo PBH: Summing HMF...", end='')

    # Conditional PS: (z, d) 
    cond_table = np.zeros((len(z_s), len(d_s)))
    dndm = hmfdata['ps_cond'] # [1 / cMpc^3 Msun]
    for i_z, z in enumerate(z_s):
        for i_d, d in enumerate(d_s):
            eff_dndm = dndm[i_z, i_d] * (m_s > m_thres)
            cond_table[i_z,i_d] = np.trapz(L_table[i_z] * eff_dndm, m_s)

    # Unconditional PS: (z)
    ps_table = np.zeros((len(zfull_s),))
    dndm = hmfdata['ps'] # [1 / cMpc^3 Msun]
    for i_z, z in enumerate(zfull_s):
        if z > z_s[-1]:
            continue
        eff_dndm = dndm[i_z] * (m_s > m_thres)
        ps_table[i_z] = np.trapz(L_table[i_z] * eff_dndm, m_s)

    # Sheth-Tormen: (z)
    st_table = np.zeros((len(zfull_s),))
    dndm = hmfdata['st'] # [1 / cMpc^3 Msun]
    for i_z, z in enumerate(zfull_s):
        if z > z_s[-1]:
            continue
        eff_dndm = dndm[i_z] * (m_s > m_thres)
        st_table[i_z] = np.trapz(L_table[i_z] * eff_dndm, m_s)
    print("Done.")

    #===== Halo PBH: Save =====
    unit_conversion = (1 * c.M_sun * c.c**2 / u.yr / u.Mpc**3).to(u.eV / u.s / u.cm**3)
    unit_conversion = unit_conversion.value

    data = {
        'cell_size' : hmfdata['cell_size'],
        'r_fixed' : hmfdata['r_fixed'],
        'mPBH' : mPBH,
        'z' : z_s,
        'zfull' : zfull_s,
        'd' : d_s,
        'ps_cond' : cond_table * unit_conversion,
        'ps' : ps_table * unit_conversion,
        'st' : st_table * unit_conversion,
        'units' : 'cell_size: [cMpc]. r_fixed: [cMpc]. mPBH: [Msun]. z/zfull: [1]. d: [1]. All rates: [eV / s / cfcm^3].',
        'shapes' : 'ps_cond: (z, d). ps, st: (zfull).',
    }
    save_h5_dict(halo_file, data)
    print("Saved halo PBH table.")


    #===== Cosmo PBH: HMF f_coll =====
    # xxx_table: [Msun/yr / cMpc^3]
    print("Cosmo PBH: Evaluating f_coll...", end='')

    L_cosmo_density_vmap = jax.jit(jax.vmap(am.L_cosmo_density, in_axes=(None, None, None, None, 0, 0)))
    L_cosmo_density_vcbavg_vmap = jax.jit(jax.vmap(am.L_cosmo_density_vcbavg, in_axes=(None, None, None, None, 0)))
    
    def get_rho_dm_rho_b(rho_coll, z, d=0.):
        """Returns rho_dm [Msun / cMpc^3] and rho_b [g/cm^3].
        
        Args:
            rho_coll (float): Collapsed mass density [Msun / cMpc^3]
            z (float): Redshift
            d (float): Overdensity
        """
        if d == -1:
            return 0., 0.
        rho_tot = (1 + d) * RHO_M # [Msun / cMpc^3]
        f_coll = rho_coll / rho_tot # [1]
        if f_coll >= 1:
            logging.warning(f"Warning: f_coll={f_coll} >= 1 at z={z}, d={d}. Setting to 1.")
            return 0., 0.
        rho_dm = cosmo.Odm0 / cosmo.Om0 * rho_tot * (1 - f_coll) # [Msun / cMpc^3]
        # Note: baryon density is not in conformal units!
        rho_b_avg = (cosmo.Ob(z) * cosmo.critical_density(z)).to(u.g/u.cm**3).value # [g/cm^3]
        rho_b = (1 + d) * rho_b_avg * (1 - f_coll) # [g/cm^3]
        return rho_dm, rho_b

    # Conditional PS: (z, cinf, dsub, vcb)
    cond_table = np.zeros((len(z_s), len(cinf_s), len(dsub_s), len(vcb_s)))
    dndm = hmfdata['ps_cond'] # [1 / cMpc^3 Msun]
    for i_z, z in enumerate(tqdm(z_s, desc='Conditional PS')):
        for i_d, d in enumerate(dsub_s):
            rho_coll = np.trapz(m_s * dndm[i_z,i_d], m_s) # [Msun / cMpc^3]
            rho_dm, rho_b = get_rho_dm_rho_b(rho_coll, z, d)
            c_input, v_input = jnp.meshgrid(cinf_s, vcb_s, indexing='ij')
            cond_table[i_z,:,i_d,:] = L_cosmo_density_vmap(mPBH, z, rho_dm, rho_b, c_input.flatten(), v_input.flatten()).reshape(len(cinf_s), len(vcb_s))

    # Unconditional PS: (zfull, cinf)
    ps_table = np.zeros((len(zfull_s), len(cinf_s)))
    dndm = hmfdata['ps'] # [1 / cMpc^3 Msun]
    for i_z, z in enumerate(tqdm(zfull_s, desc='Unconditional PS')):
        rho_coll = np.trapz(m_s * dndm[i_z], m_s) if z <= z_s[-1] else 0. # [Msun / cMpc^3]
        rho_dm, rho_b = get_rho_dm_rho_b(rho_coll, z)
        ps_table[i_z,:] = L_cosmo_density_vcbavg_vmap(mPBH, z, rho_dm, rho_b, cinf_s)

    # Unconditional Sheth-Tormen: (zfull, cinf)
    st_table = np.zeros((len(zfull_s), len(cinf_s)))
    dndm = hmfdata['st'] # [1 / cMpc^3 Msun]
    for i_z, z in enumerate(tqdm(zfull_s, desc='Unconditional ST')):
        rho_coll = np.trapz(m_s * dndm[i_z], m_s) if z <= z_s[-1] else 0. # [Msun / cMpc^3]
        rho_dm, rho_b = get_rho_dm_rho_b(rho_coll, z)
        st_table[i_z,:] = L_cosmo_density_vcbavg_vmap(mPBH, z, rho_dm, rho_b, cinf_s)
    print("Done.")

    #===== Save =====
    unit_conversion = (1 * c.M_sun * c.c**2 / u.yr / u.Mpc**3).to(u.eV / u.s / u.cm**3)
    unit_conversion = unit_conversion.value

    data = {
        'cell_size' : hmfdata['cell_size'],
        'r_fixed' : hmfdata['r_fixed'],
        'm_PBH' : mPBH,
        'z' : z_s,
        'zfull' : zfull_s,
        'dsub' : dsub_s,
        'cinf' : cinf_s,
        'vcb' : vcb_s,
        'ps_cond' : cond_table * unit_conversion,
        'ps' : ps_table * unit_conversion,
        'st' : st_table * unit_conversion,
        'units' : 'cell_size: [cMpc]. r_fixed: [cMpc]. m_PBH: [Msun]. z/zfull: [1]. dsub: [1]. cinf: [km/s], vcb: [km/s]. all rates: [eV / s / cfcm^3].',
        'shapes' : 'ps_cond: (z, cinf, dsub, vcb). ps, st: (zfull, cinf).',
    }
    save_h5_dict(cosmo_file, data)
    print("Saved cosmo PBH table.")