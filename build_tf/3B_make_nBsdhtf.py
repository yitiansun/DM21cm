"""Make DarkHistory transfer functions at all nBscale values for electron transfer function."""

import os
import sys
import pickle
from tqdm import tqdm
import argparse

import numpy as np
from astropy.io import fits
from scipy import interpolate

sys.path.append("..")
from dm21cm.utils import load_h5_dict
import dm21cm.physics as phys

sys.path.append(os.environ['DH_DIR'])
import darkhistory.spec.spectools as spectools
import darkhistory.physics as dh_phys


if __name__ == '__main__':

    #===== Config =====
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', type=str, help='run name')
    args = parser.parse_args()

    run_name = args.name

    data_dir = f"{os.environ['DM21CM_DATA_DIR']}/tf/{run_name}"
    save_dir = f"{os.environ['DM21CM_DATA_DIR']}/tf/{run_name}/nBsdhtf"
    os.makedirs(save_dir, exist_ok=True)

    abscs = load_h5_dict(f"../data/abscissas/abscs_{run_name}.h5")

    #===== Load =====
    hep_tf_full = np.load(f'{data_dir}/phot/hep_tf_rxneo.npy') # rxneo
    lep_tf_full = np.load(f'{data_dir}/phot/lep_tf_rxneo.npy')
    lee_tf_full = np.load(f'{data_dir}/phot/lee_tf_rxneo.npy')
    hed_tf_full = np.load(f'{data_dir}/phot/hed_tf_rxneo.npy')
    cmbloss_full = np.load(f'{data_dir}/phot/cmbloss_rxneo.npy')
    lowerbound_full = np.load(f'{data_dir}/phot/lowerbound_rxneo.npy')


    for i_nBs, nBs in enumerate(abscs['nBs']):

        print(f'i_nBs={i_nBs}: ', end='', flush=True)

        hep_tf = hep_tf_full[:, :, i_nBs, ...] # rxeo
        lep_tf = lep_tf_full[:, :, i_nBs, ...]
        lee_tf = lee_tf_full[:, :, i_nBs, ...]
        hed_tf = hed_tf_full[:, :, i_nBs, ...]
        cmbloss = cmbloss_full[:, :, i_nBs, ...]
        lowerbound = lowerbound_full[:, :, i_nBs, ...]

        dlnz = abscs['dlnz']

        #===== lowengphot reconstruction =====
        for i_rs in range(len(abscs['rs'])):
            for i_x in range(len(abscs['x'])):
                for i in range(len(abscs['photE'])):
                    if lep_tf[i_rs,i_x,i,i] > 1e-40:
                        break
                    lep_tf[i_rs,i_x,i,i] = 1.

        #===== add cmbloss to highengphot =====
        apply_redshift = False

        dlnphoteng = np.log(5565952217145.328/1e-4)/500
        rate = dlnz/dlnphoteng
        for i_rs in tqdm(range(len(abscs['rs']))):
            for i_x in range(len(abscs['x'])):
                tf = hep_tf[i_rs, i_x]
                cl = cmbloss[i_rs, i_x]
                rs = abscs['rs'][i_rs]
                lb = lowerbound[i_rs, i_x]
                i_lb = np.searchsorted(abscs['photE'], lb)
                if apply_redshift:
                    for i_in in range(i_lb, 500):
                        tf[i_in][i_in-1] += rate
                        tf[i_in][i_in]   -= rate
                cmb_un = spectools.discretize(abscs['photE'], dh_phys.CMB_spec, dh_phys.TCMB(rs)) # unnormalized
                cmb_un_E = cmb_un.toteng()
                for i_in in range(500):
                    cmb_E = cl[i_in] # [eV]
                    tf[i_in] += (-cmb_E/cmb_un_E) * cmb_un.N

        #===== modify dhtf =====
        tfgv_dict = {
            'highengphot' : hep_tf,
            'lowengphot' : lep_tf,
            'lowengelec': lee_tf,
        }

        #----- to photons -----
        for name in ['highengphot', 'lowengphot', 'lowengelec']:
            tf = pickle.load(open(os.environ['DH_DATA_DIR']+f'/{name}_tf_interp.raw', 'rb'))
            tfgv = tfgv_dict[name]

            tf.rs_nodes = np.array([49., 1600.])
            tf._log_interp = False
            tf.dlnz = [dlnz, 0.001, 0.001]
            tf.rs[0] = abscs['rs']

            xHe_s = np.array([0., 1.]) # fake xHe dependence
            xHe_grid, xH_grid = np.meshgrid(xHe_s, abscs['x'])
            tf.x[0] = np.stack([xH_grid, xHe_grid], axis=-1)

            tfgv_expanded = np.repeat([tfgv], 2, axis=0)
            tfgv_xhreo = np.einsum('hrxeo -> xhreo', tfgv_expanded)
            tf.grid_vals[0] = tfgv_xhreo
            tf.interp_func[0] = interpolate.RegularGridInterpolator(
                (abscs['x'], xHe_s, abscs['rs']), tf.grid_vals[0]
            )
            
            pickle.dump(tf, open(f'{save_dir}/{name}_dhtf_nBs{i_nBs}.p', 'wb'))
            print(name, end=' ', flush=True)

        #----- to depositions -----
        tf = pickle.load(open(os.environ['DH_DATA_DIR']+'/highengdep_interp.raw', 'rb'))
        tfgv = hed_tf

        tf.rs_nodes = np.array([49., 1600.])
        tf._log_interp = False
        tf.dlnz = [dlnz, 0.001, 0.001]
        tf.rs[0] = abscs['rs']

        xHe_s = np.array([0., 1.])
        xHe_grid, xH_grid = np.meshgrid(xHe_s, abscs['x'])
        tf.x[0] = np.stack([xH_grid, xHe_grid], axis=-1)

        tfgv_expanded = np.repeat([tfgv], 2, axis=0)
        tfgv_xhreo = np.einsum('hrxeo -> xhreo', tfgv_expanded)

        for i, rs in enumerate(abscs['rs']):
            dt = phys.dt_step(rs-1, np.exp(dlnz))
            # dt_dh = (dlnz/dh_phys.hubble(rs)) # DH dt
            # dt = dts[i, 1] # IDL dt
            tfgv_xhreo[:, :, i, :, :] /= dt
        tf.grid_vals[0] = tfgv_xhreo
        tf.interp_func[0] = interpolate.RegularGridInterpolator(
            (abscs['x'], xHe_s, abscs['rs']), tf.grid_vals[0]
        )

        pickle.dump(tf, open(f'{save_dir}/highengdep_dhtf_nBs{i_nBs}.p', 'wb'))
        print('hed.', flush=True)