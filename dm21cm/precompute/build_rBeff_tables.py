"""Build rBeff table for PBH accretion."""

import os
import sys
import numpy as np
from scipy import interpolate
from tqdm import tqdm

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

WDIR = os.environ['DM21CM_DIR']
sys.path.append(WDIR)
from dm21cm.precompute.accretion import veff_HALO
from dm21cm.utils import save_h5_dict


if __name__ == '__main__':

    mPBH_s = jnp.array([1e0, 1e1, 1e2, 1e3, 1e4]) # [mPBH_sun]
    z_s = jnp.geomspace(4, 3001, 300)
    r_s = jnp.geomspace(1e-5, 1e20, 300) # [km]
    veff_s = jnp.geomspace(1e0, 1e8, 300) # [km/s]

    save_dir = f"{WDIR}/data/pbh-accretion"
    os.makedirs(save_dir, exist_ok=True)

    table_mzv = np.zeros((len(mPBH_s), len(z_s), len(veff_s)))
    for i_m, m in enumerate(tqdm(mPBH_s)):
        for i_z, z in enumerate(z_s):
            v_s = veff_HALO(z=z, M=m, rBeff=r_s)
            interp = interpolate.interp1d(v_s, r_s, bounds_error=False, fill_value='extrapolate')
            table_mzv[i_m, i_z] = interp(veff_s)

    data = {
        'mPBH' : mPBH_s,
        'z' : z_s,
        'veff' : veff_s,
        'table' : table_mzv,
        'units' : 'mPBH: [mPBH_sun], z: [1], veff: [km/s], r: [km].',
        'shapes' : 'table: (mPBH, z, veff).',
    }
    save_h5_dict(f"{save_dir}/rBeff_mzv.h5", data)