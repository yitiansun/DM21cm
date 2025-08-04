"""Build pwave table."""
# This script is fast with GPUs.

import os
import sys

import numpy as np
import astropy.units as u
import astropy.constants as c
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from tqdm import tqdm

WDIR = os.environ['DM21CM_DIR']
sys.path.append(WDIR)
from dm21cm.utils import save_h5_dict, load_h5_dict
from dm21cm.precompute.halo import DM_FRAC, cmz, rel_v_disp, nfw_density, nfw_info


if __name__ == '__main__':

    #===== setting =====
    m_high_cutoff = 1e11 # [Msun]
    filename = 'pwave_hmf_summed_rate_mc1e11.h5'

    #===== Initialization =====
    hmfdata = load_h5_dict(f"{WDIR}/data/hmf/hmf.h5")
    z_s = hmfdata['z'] # [1]    | redshift
    d_s = hmfdata['d'] # [1]    | delta (overdensity)
    m_s = hmfdata['m'] # [Msun] | halo mass

    # extend z to zero out annihilation rate at higher redshifts
    z_max = z_s[-1]
    zext_s = np.concatenate((z_s, [z_max+1e-6, 4000]))

    save_dir = f"{WDIR}/data/production"
    os.makedirs(save_dir, exist_ok=True)

    #===== Annihilation rate table =====
    # (z, m)
    C = c.c.to(u.pc/u.s).value # [pc / s]
    rel_v_disp_vmap = jax.vmap(rel_v_disp, in_axes=(0, None, None, None))

    @jax.jit
    # @partial(jax.vmap, in_axes=(0, 0, None)) # can't vmap. memory requirement too high.
    def partial_ann_rate(m, c, z):
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
            m (float): Mass of the halo [Msun]
            c (float): Concentration parameter
            z (float): Redshift
        """
        rho_s, r_s, r_delta = nfw_info(m, c, z) # [Msun / pc^3], [pc], [pc]
        r_arr = jnp.geomspace(1e-4*r_s, r_delta, 1001) # [pc]
        rel_v_disp_arr = jnp.nan_to_num(rel_v_disp_vmap(r_arr, rho_s, r_s, r_delta) / C**2) # [1]
        integrand = 4*np.pi * r_arr**2 * rel_v_disp_arr * nfw_density(r_arr, rho_s, r_s)**2 # [Msun^2 / pc^4]
        return 1/2 * DM_FRAC**2 * jnp.trapz(integrand, r_arr)

    ann_rates = np.zeros((len(z_s), len(m_s)))

    for i_z, z in enumerate(tqdm(z_s, desc='Annihilation rate')):
        c_s = cmz(m_s, z)
        for i_m, m in enumerate(m_s):
            ann_rates[i_z, i_m] = partial_ann_rate(m, c_s[i_m], z) # [Msun^2 / pc^3]


    #===== Summing HMF =====
    # Conditional PS
    cond_table = np.zeros((len(zext_s), len(d_s)))
    dndm = hmfdata['ps_cond'] # [1 / cMpc^3 Msun]
    for i_z, z in enumerate(tqdm(zext_s, desc='Conditional PS')):
        if z > z_max:
            continue
        for i_d, d in enumerate(d_s):
            dndm_zd = dndm[i_z, i_d] * (m_s <= m_high_cutoff)
            cond_table[i_z,i_d] = np.trapz(ann_rates[i_z] * dndm_zd, m_s) # [Msun^2/pc^3 / cMpc^3]

    # Unconditional PS
    ps_table = np.zeros((len(zext_s),))
    dndm = hmfdata['ps'] # [1 / cMpc^3 Msun]
    for i_z, z in enumerate(tqdm(zext_s, desc='Unconditional PS')):
        if z > z_max:
            continue
        dndm_z = dndm[i_z] * (m_s <= m_high_cutoff)
        ps_table[i_z] = np.trapz(ann_rates[i_z] * dndm_z, m_s) # [Msun^2/pc^3 / cMpc^3]

    # Sheth-Tormen
    st_table = np.zeros((len(zext_s),))
    dndm = hmfdata['st'] # [1 / cMpc^3 Msun]
    for i_z, z in enumerate(tqdm(zext_s, desc='Unconditional ST')):
        if z > z_max:
            continue
        dndm_z = dndm[i_z] * (m_s <= m_high_cutoff)
        st_table[i_z] = np.trapz(ann_rates[i_z] * dndm_z, m_s) # [Msun^2/pc^3 / cMpc^3]


    #===== Save =====
    # table units = [Msun^2/pc^3 / cMpc^3]
    unit_conversion = (1 * c.M_sun**2 * c.c**4 / (u.pc**3 * u.Mpc**3)).to(u.eV**2 / u.cm**3 / u.cm**3)
    unit_conversion = unit_conversion.value

    data = {
        'cell_size' : hmfdata['cell_size'],
        'r_fixed' : hmfdata['r_fixed'],
        'z' : zext_s,
        'd' : d_s,
        'ps_cond' : cond_table * unit_conversion,
        'ps' : ps_table * unit_conversion,
        'st' : st_table * unit_conversion,
        'units' : 'cell_size: [cMpc]. r_fixed: [cMpc]. z: [1]. d: [1]. All rates: [eV^2 / cm^3 / cfcm^3].',
        'shapes' : 'ps_cond: (z, d). ps, st: (z,).',
    }
    save_h5_dict(save_dir + "/" + filename, data)