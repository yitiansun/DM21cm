import os
import sys

import numpy as np
import astropy.units as u
import astropy.constants as c
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import halomod
from tqdm import tqdm

sys.path.append(os.environ['DM21CM_DIR'])
from dm21cm.preprocessing.halo import DM_FRAC, fix_cmz_numerical_issues, rel_v_disp, nfw_density, nfw_info
from dm21cm.utils import save_h5_dict


if __name__ == '__main__':

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

    for i, z in enumerate(tqdm(z_range)):

        hm = halomod.DMHaloModel(
            halo_concentration_model='Ludlow16', z=z_range[i],
            Mmin=log10_m_min, Mmax=log10_m_max, dlog10m=dlog10m,
            mdef_model='SOCritical', halo_profile_model=halomod.profiles.NFW
        )
        m, cmz = fix_cmz_numerical_issues(hm.m, hm.cmz_relation)
        for j in range(len(hm.m)):
            annihilation_rates[i, j] = partial_ann_rate(m[j], cmz[j], z)

    #===== Save =====
    data = {
        'z': z_range,
        'm': m_range,
        'ann_rate': annihilation_rates,
        'units' : 'm: [Msun]. ann_rate: [Msun^2 / pc^3].'
    }
    save_h5_dict('../../data/pwave/pwave_partial_ann_rate_zm.h5', data)