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
from dm21cm.preprocessing.halo import DM_FRAC, cmz, rel_v_disp, nfw_density, nfw_info
from dm21cm.preprocessing.hmfe import SigmaMInterpSphere, HMFEvaluator, SphereWindow
from dm21cm.utils import save_h5_dict, load_h5_dict


if __name__ == '__main__':

    #===== Initialization =====
    smi = SigmaMInterpSphere(res=4001)
    m_s = smi.m_s                      # [Msun] | halo mass
    z_s = np.linspace(0, 50, 51)       # [1]    | redshift
    d_s = jnp.linspace(-1, 1.5, 128)   # [1]    | delta (overdensity)
    r_s = jnp.geomspace(0.1, 512, 128) # [cMpc] | radius

    assert len(sys.argv) >= 2

    if sys.argv[1] == 'halo_rate_zm': # annihilation rate of single halos

        #===== Calculation =====
        C = c.c.to(u.pc/u.s).value # [pc / s]
        rel_v_disp_vmap = jax.vmap(rel_v_disp, in_axes=(0, None, None, None))

        @jax.jit
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

        for i_z, z in enumerate(tqdm(z_s)):
            c_s = cmz(m_s, z)
            for i_m, m in enumerate(m_s):
                ann_rates[i_z, i_m] = partial_ann_rate(m, cmz[i_m], z)

        #===== Save =====
        ann_data = {
            'z': z_s,
            'm': m_s,
            'ann_rate': ann_rates,
            'units' : 'z: [1]. m: [Msun]. ann_rate: [Msun^2 / pc^3].'
        }
        save_h5_dict('../../data/pwave/pwave_halo_rate_zm.h5', ann_data)

    elif sys.argv[1] == 'sum': # Sum over HMF. Dimensions: PS cond: (r, z, d), PS uncond (z,), (z,).

        #===== Initialization =====
        ann_data = load_h5_dict('../../data/pwave/pwave_halo_rate_zm.h5')
        ann_rates = ann_data['ann_rate'] # [Msun^2 / pc^3]
        hmfe = HMFEvaluator(smi)
        ws = SphereWindow()

        #===== Conditional Press-Schechter =====
        cond_table = np.zeros((len(z_s), len(d_s), len(r_s)))

        @jax.jit
        @partial(jax.vmap, in_axes=(0, None, None))
        def cond_dndm_func(r, d, z):
            return jnp.nan_to_num(hmfe.dNdM_Conditional(ws.RtoM(r), d, z))

        for i_z, z in enumerate(tqdm(z_s, desc='Conditional PS')):
            for i_d, d in enumerate(d_s):
                cond_dndm_at_zd = cond_dndm_func(r_s, d, z) # [1 / cMpc^3 Msun]
                for i_r, r in enumerate(r_s):
                    cond_table[i_z,i_d,i_r] = np.trapz(ann_rates[i_z] * cond_dndm_at_zd[i_r], m_s) # [Msun^2 / cMpc^3 / pc^3]
        cond_table = np.einsum('zdr->rzd', cond_table) # transpose for convenience

        #===== Unconditional Press-Schechter =====
        ps_table = np.zeros((len(z_s),))

        for i_z, z in enumerate(tqdm(z_s, desc='Unconditional PS')):
            ps_table[i_z] = np.trapz(ann_rates[i_z] * hmfe.dNdM(m_s, z), m_s) # [Msun^2 / cMpc^3 / pc^3]

        #===== Sheth-Tormen =====
        st_table = np.zeros((len(z_s),))

        for i_z, z in enumerate(tqdm(z_s, desc='Unconditional ST')):
            st_table[i_z] = np.trapz(ann_rates[i_z] * hmfe.dNdM_ST(m_s, z), m_s) # [Msun^2 / cMpc^3 / pc^3]

        #===== Save =====
        # table units = [Msun^2 / pc^3 / cMpc^3]
        unit_conversion = (1 * c.M_sun**2 * c.c**4 / (u.pc**3 * u.Mpc**3)).to(u.eV**2 / u.cm**3 / u.cm**3)
        unit_conversion = unit_conversion.value

        data = {
            'z' : z_s,
            'd' : d_s,
            'r' : r_s,
            'ps_cond' : cond_table * unit_conversion,
            'ps' : ps_table * unit_conversion,
            'st' : st_table * unit_conversion,
            'units' : 'z: [1]. d: [1]. r: [cfMpc]. all rates: [eV^2 / cm^3 / cfcm^3].',
        }
        save_dirs = [
            "/n/home07/yitians/dm21cm/DM21cm/data/pwave",
            "/n/holylabs/LABS/iaifi_lab/Users/yitians/dm21cm/data/zf002/data",
        ]
        for save_dir in save_dirs:
            save_h5_dict(save_dir + f"/pwave_summed_rate_{smi.name}.h5", data)

    else:

        raise ValueError(sys.argv[1])