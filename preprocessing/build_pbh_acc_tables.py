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
# from functools import partial
from tqdm import tqdm
import h5py
import logging

sys.path.append(os.environ['DM21CM_DIR'])
from dm21cm.utils import save_h5_dict, load_h5_dict
from preprocessing.accretion import PBHAccretionModel
from preprocessing.halo import cmz
from preprocessing.hmfe import RHO_M


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str)
    parser.add_argument('--save_memory', action='store_true', default=False)
    args = parser.parse_args()

    #===== Initialization =====
    hmfdata = load_h5_dict("/n/holystore01/LABS/iaifi_lab/Users/yitians/dm21cm/data/hmf/hmf.h5")
    z_s = hmfdata['z'] # [1]    | redshift
    d_s = hmfdata['d'] # [1]    | delta (overdensity)
    m_s = hmfdata['m'] # [Msun] | halo mass

    zfull_s = np.concatenate((z_s, np.geomspace(51, 3000, 300)))# for Unconditional PS and ST tables
    mPBH_s = np.logspace(0, 4, 5)  # [Msun] | mass of PBH
    cinf_s = np.geomspace(
        np.sqrt(5/3 * (1+0.) * 10  * u.K * c.k_B / c.m_p).to(u.km/u.s).value, # 0.371 km/s
        np.sqrt(5/3 * (1+1.) * 1e4 * u.K * c.k_B / c.m_p).to(u.km/u.s).value, # 16.6 km/s
        128
    ) # should cover redshift 5-3000

    model_kwargs_dict = {
        'PRc10' : dict(accretion_type='PR-ADAF', c_in=10),
        'PRc23' : dict(accretion_type='PR-ADAF', c_in=23),
        'PRc30' : dict(accretion_type='PR-ADAF', c_in=30),
        'BHLl1e+00' : dict(accretion_type='BHL-ADAF', lambda_fudge=1),
        'BHLl1e-02' : dict(accretion_type='BHL-ADAF', lambda_fudge=1e-2),
    }
    am = PBHAccretionModel(**model_kwargs_dict[args.model])

    #===== Halo luminosity table =====
    # L_table: (mPBH, z, m) [Msun / yr]
    cache_file = f"../../data/pbh-accretion/L_table_{am.name}.npy"
    if os.path.exists(cache_file):
        L_table = np.load(cache_file)
        print(f"Using cached L_table_{am.name}.")
    else:
        print(f"Cache not found. Calculating L_table_{am.name}...")
        L_table = np.zeros((len(mPBH_s), len(z_s), len(m_s)))
            
        L_halo_vmap = jax.jit(jax.vmap(am.L_halo, in_axes=(None, 0, 0, None)))
        L_halo_jit = jax.jit(am.L_halo)
        for i_mPBH, mPBH in enumerate(mPBH_s):
            for i_z, z in enumerate(tqdm(z_s, desc=f'm_PBH={mPBH:.1e}')):
                m_halo_s = jnp.asarray(m_s) # [Msun]
                c_halo_s = jnp.asarray(cmz(m_s, z))
                if args.save_memory:
                    for i_m, m in enumerate(m_s):
                        L_table[i_mPBH, i_z, i_m] = L_halo_jit(mPBH, m, c_halo_s[i_m], z)
                else:
                    L_table[i_mPBH, i_z] = L_halo_vmap(mPBH, m_halo_s, c_halo_s, z)
        np.save(cache_file, L_table)


    #===== Halo PBH: Summing HMF =====
    # xxx_table: [Msun/yr / cMpc^3]
    print("Halo PBH: Summing HMF...", end='')

    # Conditional PS: (mPBH, z, d) 
    cond_table = np.zeros((len(mPBH_s), len(z_s), len(d_s)))
    dndm = hmfdata['ps_cond'] # [1 / cMpc^3 Msun]
    for i_mPBH, mPBH in enumerate(mPBH_s):
        for i_z, z in enumerate(z_s):
            for i_d, d in enumerate(d_s):
                cond_table[i_mPBH,i_z,i_d] = np.trapz(L_table[i_mPBH, i_z] * dndm[i_z, i_d], m_s)

    # Unconditional PS: (mPBH, z)
    ps_table = np.zeros((len(mPBH_s), len(zfull_s)))
    dndm = hmfdata['ps'] # [1 / cMpc^3 Msun]
    for i_mPBH, mPBH in enumerate(mPBH_s):
        for i_z, z in enumerate(zfull_s):
            ps_table[i_mPBH,i_z] = np.trapz(L_table[i_mPBH, i_z] * dndm[i_z], m_s) if z <= z_s[-1] else 0.

    # Sheth-Tormen: (mPBH, z)
    st_table = np.zeros((len(mPBH_s), len(zfull_s)))
    dndm = hmfdata['st'] # [1 / cMpc^3 Msun]
    for i_mPBH, mPBH in enumerate(mPBH_s):
        for i_z, z in enumerate(zfull_s):
            st_table[i_mPBH,i_z] = np.trapz(L_table[i_mPBH, i_z] * dndm[i_z], m_s) if z <= z_s[-1] else 0.
    print("Done.")

    #===== Halo PBH: Save =====
    unit_conversion = (1 * c.M_sun * c.c**2 / u.yr / u.Mpc**3).to(u.eV / u.s / u.cm**3)
    unit_conversion = unit_conversion.value

    data = {
        'cell_size' : hmfdata['cell_size'],
        'r_fixed' : hmfdata['r_fixed'],
        'mPBH' : mPBH_s,
        'z' : z_s,
        'zfull' : zfull_s,
        'd' : d_s,
        'ps_cond' : cond_table * unit_conversion,
        'ps' : ps_table * unit_conversion,
        'st' : st_table * unit_conversion,
        'units' : 'cell_size: [cMpc]. r_fixed: [cMpc]. mPBH: [Msun]. z/zfull: [1]. d: [1]. All rates: [eV / s / cfcm^3].',
        'shapes' : 'ps_cond: (mPBH, z, d). ps, st: (mPBH, zfull).',
    }
    save_h5_dict(f"../../data/production/pbhacc_halo_hmf_summed_rate_{am.name}.h5", data)
    print("Halo PBH: Tables saved.")


    #===== Cosmo PBH: HMF f_coll =====
    # xxx_table: [Msun/yr / cMpc^3]
    print("Cosmo PBH: Evaluating f_coll...", end='')

    # Function preparation
    L_cosmo_density_vmap = jax.jit(jax.vmap(am.L_cosmo_density, in_axes=(0, None, None, None, 0)))
    mPBH_in, cinf_in = jnp.meshgrid(mPBH_s, cinf_s, indexing='ij')
    mPBH_in = mPBH_in.flatten()
    cinf_in = cinf_in.flatten()
    def L_cosmo_density_wrapper(z, rho_dm, rho_b):
        return L_cosmo_density_vmap(mPBH_in, z, rho_dm, rho_b, cinf_in).reshape((len(mPBH_s), len(cinf_s)))
    
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

    # Conditional PS: (mPBH, z, cinf, d)
    cond_table = np.zeros((len(mPBH_s), len(z_s), len(cinf_s), len(d_s)))
    dndm = hmfdata['ps_cond'] # [1 / cMpc^3 Msun]
    for i_z, z in enumerate(tqdm(z_s, desc='Conditional PS')):
        for i_d, d in enumerate(d_s):
            rho_coll = np.trapz(m_s * dndm[i_z,i_d], m_s) # [Msun / cMpc^3]
            rho_dm, rho_b = get_rho_dm_rho_b(rho_coll, z, d)
            cond_table[:,i_z,:,i_d] = L_cosmo_density_wrapper(z, rho_dm, rho_b)

    # Unconditional PS: (mPBH, zfull, cinf)
    ps_table = np.zeros((len(mPBH_s), len(zfull_s), len(cinf_s)))
    dndm = hmfdata['ps'] # [1 / cMpc^3 Msun]
    for i_z, z in enumerate(tqdm(zfull_s, desc='Unconditional PS')):
        rho_coll = np.trapz(m_s * dndm[i_z], m_s) if z <= z_s[-1] else 0. # [Msun / cMpc^3]
        rho_dm, rho_b = get_rho_dm_rho_b(rho_coll, z)
        ps_table[:,i_z,:] = L_cosmo_density_wrapper(z, rho_dm, rho_b)

    # Unconditional Sheth-Tormen: (mPBH, zfull, cinf)
    st_table = np.zeros((len(mPBH_s), len(zfull_s), len(cinf_s)))
    dndm = hmfdata['st'] # [1 / cMpc^3 Msun]
    for i_z, z in enumerate(tqdm(zfull_s, desc='Unconditional ST')):
        rho_coll = np.trapz(m_s * dndm[i_z], m_s) if z <= z_s[-1] else 0. # [Msun / cMpc^3]
        rho_dm, rho_b = get_rho_dm_rho_b(rho_coll, z)
        st_table[:,i_z,:] = L_cosmo_density_wrapper(z, rho_dm, rho_b)
    print("Done.")

    #===== Save =====
    unit_conversion = (1 * c.M_sun * c.c**2 / u.yr / u.Mpc**3).to(u.eV / u.s / u.cm**3)
    unit_conversion = unit_conversion.value

    data = {
        'cell_size' : hmfdata['cell_size'],
        'r_fixed' : hmfdata['r_fixed'],
        'm_PBH' : mPBH_s,
        'z' : z_s,
        'zfull' : zfull_s,
        'd' : d_s,
        'cinf' : cinf_s,
        'ps_cond' : cond_table * unit_conversion,
        'ps' : ps_table * unit_conversion,
        'st' : st_table * unit_conversion,
        'units' : 'cell_size: [cMpc]. r_fixed: [cMpc]. m_PBH: [Msun]. z/zfull: [1]. d: [1]. cinf: [km/s]. all rates: [eV / s / cfcm^3].',
        'shapes' : 'ps_cond: (mPBH, z, cinf, d). ps, st: (mPBH, zfull, cinf).',
    }
    save_h5_dict(f"../../data/production/pbhacc_cosmo_rate_{am.name}.h5", data)
    print("Saved cosmo PBH tables.")