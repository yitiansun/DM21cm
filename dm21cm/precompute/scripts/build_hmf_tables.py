"""Build hmf table."""
# This script is fast with CPUs.

import os
import sys
import logging

import numpy as np
import astropy.units as u
import astropy.constants as c
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from functools import partial
from tqdm import tqdm

WDIR = os.environ['DM21CM_DIR']
sys.path.append(WDIR)
from dm21cm.utils import save_h5_dict
from dm21cm.precompute.ps import SigmaMInterpSphere, HMFEvaluator, SphereWindow


if __name__ == '__main__':

    #===== Initialization =====
    smi = SigmaMInterpSphere(res=4001)
    hmfe = HMFEvaluator(smi)
    ws = SphereWindow()

    m_s = smi.m_s                      # [Msun] | halo mass
    z_s = np.geomspace(4, 100, 300)    # [1]    | redshift
    d_s = jnp.linspace(-1, 1.6, 300)   # [1]    | delta (overdensity)

    generate_r_table = False
    if generate_r_table:
        r_s = jnp.geomspace(0.1, 10, 300)  # [cMpc] | radius

    cell_size = 2 # [cMpc] | comoving box size
    r_fixed = cell_size / np.cbrt(4*np.pi/3) # [cMpc] | r of sphere with volume cell_size^3
    m_fixed = ws.RtoM(r_fixed) # [Msun] | m of sphere with radius r_fixed

    save_dir = f"{WDIR}/data/hmf"
    os.makedirs(save_dir, exist_ok=True)


    #===== Conditional Press-Schechter =====
    if generate_r_table:
        cond_table = np.zeros((len(z_s), len(d_s), len(r_s), len(m_s)))

        @jax.jit
        @partial(jax.vmap, in_axes=(0, None, None))
        def cond_dndm_func(r, d, z):
            return jnp.nan_to_num(hmfe.dNdM_Conditional(ws.RtoM(r), d, z))

        for i_z, z in enumerate(tqdm(z_s, desc='Conditional PS')):
            for i_d, d in enumerate(d_s):
                cond_table[i_z,i_d] = cond_dndm_func(r_s, d, z) # [1 / cMpc^3 Msun]
        cond_table = np.einsum('zdrm->rzdm', cond_table)

    #===== Conditional Press-Schechter at Fixed R =====
    cond_fixedr_table = np.zeros((len(z_s), len(d_s), len(m_s)))

    @jax.jit
    def cond_fixedr_dndm_func(d, z):
        return jnp.nan_to_num(hmfe.dNdM_Conditional(m_fixed, d, z))

    for i_z, z in enumerate(tqdm(z_s, desc='Conditional PS fixed r')):
        for i_d, d in enumerate(d_s):
            cond_fixedr_table[i_z,i_d] = cond_fixedr_dndm_func(d, z) # [1 / cMpc^3 Msun]

    #===== Unconditional Press-Schechter =====
    ps_table = np.zeros((len(z_s), len(m_s)))

    for i_z, z in enumerate(tqdm(z_s, desc='Unconditional PS')):
        ps_table[i_z] = hmfe.dNdM(m_s, z)

    #===== Sheth-Tormen =====
    st_table = np.zeros((len(z_s), len(m_s)))

    for i_z, z in enumerate(tqdm(z_s, desc='Unconditional ST')):
        st_table[i_z] = hmfe.dNdM_ST(m_s, z)

    #===== Save =====
    if generate_r_table:
        if np.any(np.isnan(cond_table)):
            logging.warning("NaNs in ps_cond. Check for errors.")
        # ps_cond with varying r
        data = {
            'z' : z_s,
            'd' : d_s,
            'r' : r_s,
            'm' : m_s,
            'ps_cond' : cond_table,
            'shapes' : 'ps_cond: (z, d, r, m).',
            'units' : 'z: [1]. d: [1]. r: [cfMpc]. m: [Msun]. All tables: [1 / cMpc^3 Msun].',
        }
        save_h5_dict(save_dir + "/hmf_r.h5", data)

    # ps_cond with fixed r and other tables
    for name, t in zip(['ps_cond_fixedr', 'ps', 'st'], [cond_fixedr_table, ps_table, st_table]):
        if np.any(np.isnan(t)):
            logging.warning(f"NaNs in {name}. Check for errors.")
    data = {
        'cell_size' : cell_size,
        'r_fixed' : r_fixed,
        'z' : z_s,
        'd' : d_s,
        'm' : m_s,
        'ps_cond' : cond_fixedr_table,
        'ps' : ps_table,
        'st' : st_table,
        'shapes' : 'ps_cond: (z, d, m). ps, st: (z, m).',
        'units' : 'cell_size: [cfMpc]. r_fixed: [cfMpc]. z: [1]. d: [1]. m: [Msun]. All tables: [1 / cMpc^3 Msun].',
    }
    save_h5_dict(save_dir + "/hmf.h5", data)