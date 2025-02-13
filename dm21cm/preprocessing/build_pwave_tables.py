import os
import sys

import numpy as np
from scipy import interpolate
import astropy.units as u
import astropy.constants as c
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from functools import partial

import halomod
from tqdm import tqdm

sys.path.append(os.environ['DM21CM_DIR'])
from dm21cm.preprocessing.halo import DM_FRAC, fix_cmz_numerical_issues, rel_v_disp, nfw_density, nfw_info
from dm21cm.preprocessing.hmfe import SigmaMInterpSphere, HMFEvaluator, SphereWindow
from dm21cm.utils import save_h5_dict, load_h5_dict


if __name__ == '__main__':

    assert len(sys.argv) >= 2

    if sys.argv[1] == 'ann_rate':

        #===== Initialization =====
        log10_m_min = 0.
        log10_m_max = 19.
        dlog10m = 0.025
        m_range = np.logspace(log10_m_min, log10_m_max, int((log10_m_max-log10_m_min)/dlog10m + 1))[:-1]
        z_range = np.linspace(0, 50, 51)

        annihilation_rates = np.zeros((len(z_range), len(m_range)))

        #===== Calculation =====
        C = c.c.to(u.pc/u.s).value # [pc / s]
        rel_v_disp_vmap = jax.vmap(rel_v_disp, in_axes=(0, None, None, None))

        @jax.jit
        def partial_ann_rate(M, c, z):
            """Returns partial annihilation rate dNpartial/dt [Msun^2 / pc^3]
            with dNpartial/dt = 1/2 * DM_FRAC**2 * Int[dV <v_rel^2>/c^2 rho^2]

            Notes:
                Annihilation rate in a halo:
                    dN/dt = 1/2 * DM_FRAC**2 * (1/m_DM)**2 * Int[dV <sigma v> rho^2]
                P-wave annihilation:
                    <sigma v> = C_sigma * <v_rel^2> / c^2
                Therefore the full annihilation rate of a halo is:
                    dN/dt = C_sigma / m_DM**2 * dNpartial/dt
                Units:
                    [1/T] = [L^3/T] / [M^2] * [M^2/L^3]
                    C_sigma / m_DM**2 needs to have units [pc^3 / Msun^2 / T]
            
            Args:
                M (float): Mass of the halo [M_sun]
                c (float): Concentration parameter
                z (float): Redshift
            """
            rho_s, r_s, r_delta = nfw_info(M, c, z) # [Msun / pc^3], [pc], [pc]
            r_arr = jnp.geomspace(1e-4*r_s, r_delta, 1001) # [pc]
            rel_v_disp_arr = jnp.nan_to_num(rel_v_disp_vmap(r_arr, rho_s, r_s, r_delta) / C**2) # [1]
            integrand = 4*np.pi * r_arr**2 * rel_v_disp_arr * nfw_density(r_arr, rho_s, r_s)**2 # [Msun^2 / pc^4]
            return 1/2 * DM_FRAC**2 * jnp.trapz(integrand, r_arr)

        for i_z, z in enumerate(tqdm(z_range)):

            hm = halomod.DMHaloModel(
                halo_concentration_model='Ludlow16', z=z_range[i_z],
                Mmin=log10_m_min, Mmax=log10_m_max, dlog10m=dlog10m,
                mdef_model='SOCritical', halo_profile_model=halomod.profiles.NFW
            )
            m, cmz = fix_cmz_numerical_issues(hm.m, hm.cmz_relation)
            for i_m in range(len(hm.m)):
                annihilation_rates[i_z, i_m] = partial_ann_rate(m[i_m], cmz[i_m], z)

        #===== Save =====
        ann_data = {
            'z': z_range,
            'm': m_range,
            'ann_rate': annihilation_rates,
            'units' : 'm: [Msun]. ann_rate: [Msun^2 / pc^3].'
        }
        save_h5_dict('../../data/pwave/pwave_partial_ann_rate_zm.h5', ann_data)

    elif sys.argv[1] == 'sum':

        #===== Initialization =====
        ann_data = load_h5_dict('../../data/pwave/pwave_partial_ann_rate_zm.h5')
        smi = SigmaMInterpSphere()

        log10_interp = interpolate.RegularGridInterpolator(
            (ann_data['z'], np.log10(ann_data['m'])), np.log10(ann_data['ann_rate']),
            bounds_error=False, fill_value=np.min(np.log10(ann_data['ann_rate']))
        )

        z_s = jnp.array(ann_data['z']) # [1]
        m_s = smi.m_s # [Msun]
        ann_rate_table = np.zeros((len(z_s), len(m_s)))
        for i_z, z in enumerate(ann_data['z']):
            zm_in = np.stack([np.full(m_s.shape, z), np.log10(m_s)], axis=-1)
            ann_rate_table[i_z] = 10**log10_interp(zm_in)

        d_s = jnp.linspace(-1, 1.5, 128) # [1] | overdensity
        r_s = jnp.geomspace(0.1, 512, 128) # [cMpc]

        hmfe = HMFEvaluator(smi)
        ws = SphereWindow()

        #===== Conditional Press-Schechter =====
        cond_ann_table = np.zeros((len(z_s), len(d_s), len(r_s)))

        @jax.jit
        @partial(jax.vmap, in_axes=(0, None, None))
        def cond_dndm_func(r, d, z):
            return jnp.nan_to_num(hmfe.dNdM_Conditional(ws.RtoM(r), d, z))

        for i_z, z in enumerate(tqdm(z_s, desc='Conditional PS')):
            for i_d, d in enumerate(d_s):
                cond_dndm_at_zd = cond_dndm_func(r_s, d, z) # [1 / cMpc^3 Msun]
                for i_r, r in enumerate(r_s):
                    cond_ann_table[i_z,i_d,i_r] = np.trapz(ann_rate_table[i_z] * cond_dndm_at_zd[i_r], m_s) # [Msun^2 / cMpc^3 / pc^3]
        cond_ann_table = np.einsum('zdr->rzd', cond_ann_table)

        #===== Unconditional Press-Schechter =====
        ps_ann_table = np.zeros((len(z_s),))

        for i_z, z in enumerate(tqdm(z_s, desc='Unconditional PS')):
            ps_ann_table[i_z] = np.trapz(ann_rate_table[i_z] * hmfe.dNdM(m_s, z), m_s) # [Msun^2 / cMpc^3 / pc^3]

        #===== Sheth-Tormen =====
        st_ann_table = np.zeros((len(z_s),))

        for i_z, z in enumerate(tqdm(z_s, desc='Unconditional ST')):
            st_ann_table[i_z] = np.trapz(ann_rate_table[i_z] * hmfe.dNdM_ST(m_s, z), m_s) # [Msun^2 / cMpc^3 / pc^3]

        #===== Save =====
        Msun2_pc3Mpc3_to_eV2_cm3cm3 = (1 * c.M_sun**2 * c.c**4 / (u.pc**3 * u.Mpc**3)).to(u.eV**2 / u.cm**3 / u.cm**3)
        Msun2_pc3Mpc3_to_eV2_cm3cm3 = Msun2_pc3Mpc3_to_eV2_cm3cm3.value

        data = {
            'z' : z_s,
            'd' : d_s,
            'r' : r_s,
            'ps_cond_ann_rate' : cond_ann_table * Msun2_pc3Mpc3_to_eV2_cm3cm3,
            'ps_ann_rate' : ps_ann_table * Msun2_pc3Mpc3_to_eV2_cm3cm3,
            'st_ann_rate' : st_ann_table * Msun2_pc3Mpc3_to_eV2_cm3cm3,
            'units' : 'z: [1]. d: [1]. r: [cfMpc]. all rates: [eV^2 / cm^3 / cfcm^3].',
        }
        save_dirs = [
            "/n/home07/yitians/dm21cm/DM21cm/data/pwave",
            "/n/holylabs/LABS/iaifi_lab/Users/yitians/dm21cm/data/zf002/data",
        ]
        for save_dir in save_dirs:
            save_h5_dict(save_dir + f"/pwave_partial_ann_rate_{smi.name}.h5", data)

    else:

        raise ValueError(sys.argv[1])