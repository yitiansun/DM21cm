import os
import sys

import numpy as np
import astropy.units as u
import astropy.constants as c
from astropy.cosmology import Planck18 as cosmo
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from functools import partial
from tqdm import tqdm
import h5py

sys.path.append(os.environ['DM21CM_DIR'])
from dm21cm.preprocessing.accretion import PBHAccretionModel
from dm21cm.preprocessing.halo import cmz
from dm21cm.preprocessing.hmfe import SigmaMInterpSphere, HMFEvaluator, SphereWindow, RHO_M
from dm21cm.utils import save_h5_dict, load_h5_dict


if __name__ == '__main__':

    #===== Initialization =====
    hmfdata = h5py.File("/n/holystore01/LABS/iaifi_lab/Users/yitians/dm21cm/data/hmf/hmf.h5", 'r')
    z_s = hmfdata['z'][()] # [1]    | redshift
    d_s = hmfdata['d'][()] # [1]    | delta (overdensity)
    m_s = hmfdata['m'][()] # [Msun] | halo mass

    mPBH_s = np.logspace(0, 4, 5)      # [Msun] | mass of PBH
    T_s = jnp.geomspace(10, 1e4, 128)  # [K]    | gas temperature
    T_eV_s = (1 * u.K * c.k_B).to(u.eV).value * T_s


    #===== Halo luminosity table =====
    # (mPBH, z, m) [Msun / yr]
    cache_file = "../../data/pbh-accretion/L_table.npy"
    if os.path.exists(cache_file):
        L_table = np.load(cache_file)
        print("Using cached L_table.")
    else:
        print("Cache not found. Calculating L_table...")
        L_table = np.zeros((len(mPBH_s), len(z_s), len(m_s)))
            
        for i_mPBH, mPBH in enumerate(mPBH_s):
            am = PBHAccretionModel(m_PBH=mPBH, f_PBH=1, accretion_type='PR-ADAF', c_in=23)
            L_halo_vmap = jax.jit(jax.vmap(am.L_halo, in_axes=(0, 0, None)))

            for i_z, z in enumerate(tqdm(z_s, desc=f'm_PBH={mPBH:.1e}')):
                m_halo_s = jnp.asarray(m_s) # [Msun]
                c_halo_s = jnp.asarray(cmz(m_s, z))
                L_table[i_mPBH, i_z] = L_halo_vmap(m_halo_s, c_halo_s, z)
        np.save(cache_file, L_table)


    #===== Halo PBH: Summing HMF =====
    print("Halo PBH tables:")
    # Conditional PS: (mPBH, z, d) [Msun/yr / cMpc^3]
    cond_table = np.zeros((len(mPBH_s), len(z_s), len(d_s)))
    dndm = hmfdata['ps_cond'] # [1 / cMpc^3 Msun]
    for i_mPBH, mPBH in enumerate(tqdm(mPBH_s, desc='Conditional PS')):
        for i_z, z in enumerate(z_s):
            for i_d, d in enumerate(d_s):
                cond_table[i_mPBH,i_z,i_d] = np.trapz(L_table[i_mPBH, i_z] * dndm[i_z, i_d], m_s)

    # Unconditional PS: (mPBH, z) [Msun/yr / cMpc^3]
    ps_table = np.zeros((len(mPBH_s), len(z_s)))
    dndm = hmfdata['ps'] # [1 / cMpc^3 Msun]
    for i_mPBH, mPBH in enumerate(tqdm(mPBH_s, desc='Unconditional PS')):
        for i_z, z in enumerate(z_s):
            ps_table[i_mPBH,i_z] = np.trapz(L_table[i_mPBH, i_z] * dndm[i_z], m_s)

    # Sheth-Tormen: (mPBH, z) [Msun/yr / cMpc^3]
    st_table = np.zeros((len(mPBH_s), len(z_s)))
    dndm = hmfdata['st'] # [1 / cMpc^3 Msun]
    for i_mPBH, mPBH in enumerate(tqdm(mPBH_s, desc='Unconditional ST')):
        for i_z, z in enumerate(z_s):
            st_table[i_mPBH,i_z] = np.trapz(L_table[i_mPBH, i_z] * dndm[i_z], m_s)

    #===== Halo PBH: Save =====
    unit_conversion = (1 * c.M_sun * c.c**2 / u.yr / u.Mpc**3).to(u.eV / u.s / u.cm**3)
    unit_conversion = unit_conversion.value

    data = {
        'cell_size' : hmfdata['cell_size'],
        'r_fixed' : hmfdata['r_fixed'],
        'mPBH' : mPBH_s,
        'z' : z_s,
        'd' : d_s,
        'ps_cond' : cond_table * unit_conversion,
        'ps' : ps_table * unit_conversion,
        'st' : st_table * unit_conversion,
        'units' : 'cell_size: [cMpc]. r_fixed: [cMpc]. mPBH: [Msun]. z: [1]. d: [1]. All rates: [eV / s / cfcm^3].',
        'shapes' : 'ps_cond: (mPBH, z, d). ps, st: (mPBH, z).',
    }
    save_h5_dict("../../data/production/pbhacc_halo_hmf_summed_rate.h5", data)
    print("Saved halo PBH tables.")


    #===== Cosmo PBH: HMF f_coll =====
    print("Cosmo PBH tables:")
    # Conditional PS: (mPBH, z, d, T) [Msun/yr / cMpc^3]
    cond_table = np.zeros((len(mPBH_s), len(z_s), len(d_s), len(T_s)))
    dndm = hmfdata['ps_cond'] # [1 / cMpc^3 Msun]

    for i_z, z in enumerate(tqdm(z_s, desc='Conditional PS')):
        for i_d, d in enumerate(d_s):
            if 1 + d == 0:
                continue
            rho_coll = np.trapz(m_s * dndm[i_z, i_d], m_s) # [Msun / cMpc^3]
            rho_tot = (1 + d) * RHO_M # [Msun / cMpc^3]
            f_coll = rho_coll / rho_tot # [1]
            if f_coll > 1:
                print(f"Warning: Cond PS f_coll={f_coll} > 1 at z={z}, d={d}. Setting to 1.")
                f_coll = 1
            rho_dm = cosmo.Odm0 / cosmo.Om0 * rho_tot * (1 - f_coll) # [Msun / cMpc^3]
            # Note: baryon density is not in conformal units!
            rho_b_avg = (cosmo.Ob(z) * cosmo.critical_density(z)).to(u.g/u.cm**3).value # [g/cm^3]
            rho_b = (1 + d) * rho_b_avg * (1 - f_coll) # [g/cm^3]

            for i_mPBH, mPBH in enumerate(mPBH_s):
                am = PBHAccretionModel(m_PBH=mPBH, f_PBH=1, accretion_type='PR-ADAF', c_in=23)
                for i_T, T in enumerate(T_eV_s):
                    cond_table[i_mPBH,i_z,i_d,i_T] = am.L_cosmo_density(z, rho_dm, rho_b, T)

    # Unconditional PS: (mPBH, z, T) [Msun/yr / cMpc^3]
    ps_table = np.zeros((len(mPBH_s), len(z_s), len(T_s)))
    dndm = hmfdata['ps'] # [1 / cMpc^3 Msun]

    for i_z, z in enumerate(tqdm(z_s, desc='Unconditional PS')):
        rho_coll = np.trapz(m_s * dndm[i_z], m_s) # [Msun / cMpc^3]
        rho_tot = RHO_M # [Msun / cMpc^3]
        f_coll = rho_coll / rho_tot # [1]
        if f_coll > 1:
            print(f"Warning: Uncond PS f_coll={f_coll} > 1 at z={z}. Setting to 1.")
            f_coll = 1
        rho_dm = cosmo.Odm0 / cosmo.Om0 * rho_tot * (1 - f_coll) # [Msun / cMpc^3]
        # Note: baryon density is not in conformal units!
        rho_b_avg = (cosmo.Ob(z) * cosmo.critical_density(z)).to(u.g/u.cm**3).value # [g/cm^3]
        rho_b = rho_b_avg * (1 - f_coll) # [g/cm^3]

        for i_mPBH, mPBH in enumerate(mPBH_s):
            am = PBHAccretionModel(m_PBH=mPBH, f_PBH=1, accretion_type='PR-ADAF', c_in=23)
            for i_T, T in enumerate(T_eV_s):
                ps_table[i_mPBH,i_z,i_T] = am.L_cosmo_density(z, rho_dm, rho_b, T)

    # Unconditional Sheth-Tormen: (mPBH, z, T) [Msun/yr / cMpc^3]
    st_table = np.zeros((len(mPBH_s), len(z_s), len(T_s)))
    dndm = hmfdata['st'] # [1 / cMpc^3 Msun]

    for i_z, z in enumerate(tqdm(z_s, desc='Unconditional ST')):
        rho_coll = np.trapz(m_s * dndm[i_z], m_s) # [Msun / cMpc^3]
        rho_tot = RHO_M # [Msun / cMpc^3]
        f_coll = rho_coll / rho_tot # [1]
        if f_coll > 1:
            print(f"Warning: Uncond ST f_coll={f_coll} > 1 at z={z}. Setting to 1.")
            f_coll = 1
        rho_dm = cosmo.Odm0 / cosmo.Om0 * rho_tot * (1 - f_coll) # [Msun / cMpc^3]
        # Note: baryon density is not in conformal units!
        rho_b_avg = (cosmo.Ob(z) * cosmo.critical_density(z)).to(u.g/u.cm**3).value # [g/cm^3]
        rho_b = rho_b_avg * (1 - f_coll) # [g/cm^3]

        for i_mPBH, mPBH in enumerate(mPBH_s):
            am = PBHAccretionModel(m_PBH=mPBH, f_PBH=1, accretion_type='PR-ADAF', c_in=23)
            for i_T, T in enumerate(T_eV_s):
                st_table[i_mPBH,i_z,i_T] = am.L_cosmo_density(z, rho_dm, rho_b, T)

    #===== Save =====
    unit_conversion = (1 * c.M_sun * c.c**2 / u.yr / u.Mpc**3).to(u.eV / u.s / u.cm**3)
    unit_conversion = unit_conversion.value

    data = {
        'cell_size' : hmfdata['cell_size'],
        'r_fixed' : hmfdata['r_fixed'],
        'm_PBH' : mPBH_s,
        'z' : z_s,
        'd' : d_s,
        'T' : T_s,
        'ps_cond' : cond_table * unit_conversion,
        'ps' : ps_table * unit_conversion,
        'st' : st_table * unit_conversion,
        'units' : 'cell_size: [cMpc]. r_fixed: [cMpc]. m_PBH: [Msun]. z: [1]. d: [1]. T: [K]. all rates: [eV / s / cfcm^3].',
        'shapes' : 'ps_cond: (mPBH, z, d, T). ps, st: (mPBH, z, T).',
    }
    save_h5_dict("../../data/production/pbhacc_cosmo_rate.h5", data)
    print("Saved cosmo PBH tables.")