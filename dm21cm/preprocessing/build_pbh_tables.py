import os
import sys

import numpy as np
import astropy.units as u
import astropy.constants as c
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from functools import partial
from tqdm import tqdm

sys.path.append(os.environ['DM21CM_DIR'])
from dm21cm.preprocessing.accretion import PBHAccretionModel
from dm21cm.preprocessing.halo import cmz
from dm21cm.preprocessing.hmfe import SigmaMInterpSphere, HMFEvaluator, SphereWindow, RHO_M
from dm21cm.utils import save_h5_dict, load_h5_dict


if __name__ == '__main__':

    #===== Initialization =====
    smi = SigmaMInterpSphere(res=4001)
    m_s = smi.m_s                      # [Msun] | halo mass
    z_s = np.linspace(0, 50, 51)       # [1]    | redshift
    d_s = jnp.linspace(-1, 1.5, 128)   # [1]    | delta (overdensity)
    r_s = jnp.geomspace(0.1, 512, 128) # [cMpc] | radius
    # TODO: figure out temperature range.
    mPBH_s = np.logspace(0, 4, 5)      # [Msun] | mass of PBH

    halo_file = "../../data/pbh-accretion/pbhacc_halo_rate_zm.h5"
    
    assert len(sys.argv) >= 2

    if sys.argv[1] == 'halo_rate_zm': # L for single halos.

        L_table = np.zeros((len(mPBH_s), len(z_s), len(m_s))) # [Msun / yr]

        #===== Calculation =====
        for i_mPBH, mPBH in enumerate(mPBH_s):

            am = PBHAccretionModel(m_PBH=mPBH, f_PBH=1, accretion_type='PR-ADAF', c_in=23)
            L_halo_vmap = jax.jit(jax.vmap(am.L_halo, in_axes=(0, 0, None)))

            for i_z, z in enumerate(tqdm(z_s, desc=f'm_PBH={mPBH:.1e}')):

                m_halo_s = jnp.asarray(m_s) # [Msun]
                c_halo_s = jnp.asarray(cmz(m_s, z))
                L_table[i_mPBH, i_z] = L_halo_vmap(m_halo_s, c_halo_s, z) # [Msun / yr]

        #===== Save =====
        data = {
            'm_PBH': mPBH_s,
            'z': z_s,
            'm': m_s,
            'L': L_table,
            'units' : 'm_PBH: [Msun]. m: [Msun]. L: [Msun / yr].'
        }
        save_h5_dict(halo_file, data)


    elif sys.argv[1] == 'halo_rate_sum': # Sum over HMF. Dimensions: (r, z, d), (z,), (z,).

        #===== Initialization =====
        halo_data = load_h5_dict(halo_file)
        L_table = halo_data['L'] # [Msun / yr] | (mPBH, z, m)
        hmfe = HMFEvaluator(smi)
        ws = SphereWindow()

        #===== Conditional Press-Schechter =====
        cond_table = np.zeros((len(mPBH_s), len(z_s), len(d_s), len(r_s)))

        @jax.jit
        @partial(jax.vmap, in_axes=(0, None, None))
        def cond_dndm_func(r, d, z):
            return jnp.nan_to_num(hmfe.dNdM_Conditional(ws.RtoM(r), d, z))

        for i_z, z in enumerate(tqdm(z_s, desc='Conditional PS')):
            for i_d, d in enumerate(d_s):
                cond_dndm_at_zd = cond_dndm_func(r_s, d, z) # [1 / cMpc^3 Msun]
                for i_r, r in enumerate(r_s):
                    for i_mPBH, mPBH in enumerate(mPBH_s):
                        cond_table[i_mPBH,i_z,i_d,i_r] = np.trapz(L_table[i_mPBH, i_z] * cond_dndm_at_zd[i_r], m_s)
        cond_table = np.einsum('mzdr->mrzd', cond_table) # transpose for convenience

        #===== Unconditional Press-Schechter =====
        ps_table = np.zeros((len(mPBH_s), len(z_s)))

        for i_z, z in enumerate(tqdm(z_s, desc='Unconditional PS')):
            for i_mPBH, mPBH in enumerate(mPBH_s):
                ps_table[i_mPBH,i_z] = np.trapz(L_table[i_mPBH, i_z] * hmfe.dNdM(m_s, z), m_s)

        #===== Sheth-Tormen =====
        st_table = np.zeros((len(mPBH_s), len(z_s)))

        for i_z, z in enumerate(tqdm(z_s, desc='Unconditional ST')):
            for i_mPBH, mPBH in enumerate(mPBH_s):
                st_table[i_mPBH,i_z] = np.trapz(L_table[i_mPBH, i_z] * hmfe.dNdM_ST(m_s, z), m_s)

        #===== Save =====
        # table units = [Msun / yr / cMpc^3]
        unit_conversion = (1 * c.M_sun * c.c**2 / u.yr / u.Mpc**3).to(u.eV / u.s / u.cm**3)
        unit_conversion = unit_conversion.value

        data = {
            'm_PBH' : mPBH_s,
            'z' : z_s,
            'd' : d_s,
            'r' : r_s,
            'ps_cond' : cond_table * unit_conversion,
            'ps' : ps_table * unit_conversion,
            'st' : st_table * unit_conversion,
            'units' : 'm_PBH: [Msun]. z: [1]. d: [1]. r: [cfMpc]. all rates: [eV / s / cfcm^3].',
        }
        save_dirs = [
            "/n/home07/yitians/dm21cm/DM21cm/data/pbh-accretion",
            "/n/holylabs/LABS/iaifi_lab/Users/yitians/dm21cm/data/zf002/data",
        ]
        for save_dir in save_dirs:
            save_h5_dict(save_dir + f"/pbhacc_summed_rate_{smi.name}.h5", data)


    elif sys.argv[1] == 'cosmo_rate_z': # Unbound cosmological pbhs. Dimensions: (mPBH, z, d, T)

        #===== Initialization =====
        halo_data = load_h5_dict(halo_file)
        L_table = halo_data['L'] # [Msun / yr] | (mPBH, z, m)
        hmfe = HMFEvaluator(smi)
        ws = SphereWindow()

        cell_size = 2 # [cMpc] | comoving box size
        r_fixed = cell_size / np.cbrt(4*np.pi/3) # [cMpc] | r of sphere with volume cell_size^3
        m_fixed = ws.RtoM(r_fixed) # [Msun] | m of sphere with radius r_fixed

        #===== Conditional Press-Schechter =====
        cond_table = np.zeros((len(mPBH_s), len(z_s), len(d_s), len(r_s)))

        @jax.jit
        def cond_dndm_func(d, z):
            return jnp.nan_to_num(hmfe.dNdM_Conditional(m_fixed, d, z))

        for i_z, z in enumerate(tqdm(z_s, desc='Conditional PS')):
            for i_d, d in enumerate(d_s):
                cond_dndm_at_zd = cond_dndm_func(d, z) # [1 / cMpc^3 Msun]
        #         for i_r, r in enumerate(r_s):
        #             for i_mPBH, mPBH in enumerate(mPBH_s):
        #                 cond_table[i_mPBH,i_z,i_d,i_r] = np.trapz(L_table[i_mPBH, i_z] * cond_dndm_at_zd[i_r], m_s)
        # cond_table = np.einsum('mzdr->mrzd', cond_table) # transpose for convenience

        # #===== Unconditional Press-Schechter =====
        # ps_table = np.zeros((len(mPBH_s), len(z_s)))

        # for i_z, z in enumerate(tqdm(z_s, desc='Unconditional PS')):
        #     for i_mPBH, mPBH in enumerate(mPBH_s):
        #         ps_table[i_mPBH,i_z] = np.trapz(L_table[i_mPBH, i_z] * hmfe.dNdM(m_s, z), m_s)

        # #===== Sheth-Tormen =====
        # st_table = np.zeros((len(mPBH_s), len(z_s)))

        # for i_z, z in enumerate(tqdm(z_s, desc='Unconditional ST')):
        #     for i_mPBH, mPBH in enumerate(mPBH_s):
        #         st_table[i_mPBH,i_z] = np.trapz(L_table[i_mPBH, i_z] * hmfe.dNdM_ST(m_s, z), m_s)

        # #===== Save =====
        # # table units = [Msun / yr / cMpc^3]
        # unit_conversion = (1 * c.M_sun * c.c**2 / u.yr / u.Mpc**3).to(u.eV / u.s / u.cm**3)
        # unit_conversion = unit_conversion.value

        # data = {
        #     'm_PBH' : mPBH_s,
        #     'z' : z_s,
        #     'd' : d_s,
        #     'r' : r_s,
        #     'ps_cond' : cond_table * unit_conversion,
        #     'ps' : ps_table * unit_conversion,
        #     'st' : st_table * unit_conversion,
        #     'units' : 'm_PBH: [Msun]. z: [1]. d: [1]. r: [cfMpc]. all rates: [eV / s / cfcm^3].',
        # }
        # save_dirs = [
        #     "/n/home07/yitians/dm21cm/DM21cm/data/pbh-accretion",
        #     "/n/holylabs/LABS/iaifi_lab/Users/yitians/dm21cm/data/zf002/data",
        # ]
        # for save_dir in save_dirs:
        #     save_h5_dict(save_dir + f"/pbhacc_summed_rate_{smi.name}.h5", data)

        #===== Calculation =====
        f_coll_ps_table = np.zeros((len(z_s),))
        f_coll_st_table = np.zeros((len(z_s),))
        for i_z, z in enumerate(tqdm(z_s)):
            f_coll_ps_table[i_z] = np.trapz(m_s * hmfe.dNdM(m_s, z), m_s) / RHO_M
            f_coll_st_table[i_z] = np.trapz(m_s * hmfe.dNdM_ST(m_s, z), m_s) / RHO_M

        ps_cosmo_table = np.zeros((len(mPBH_s), len(z_s)))
        st_cosmo_table = np.zeros((len(mPBH_s), len(z_s)))
        for i_mPBH, mPBH in enumerate(mPBH_s):
            am = PBHAccretionModel(m_PBH=mPBH, f_PBH=1, accretion_type='PR-ADAF', c_in=23)
            for i_z, z in enumerate(z_s):
                ps_cosmo_table[i_mPBH,i_z] = am.L_cosmo_density(z, f_coll)

    else:

        raise ValueError(sys.argv[1])