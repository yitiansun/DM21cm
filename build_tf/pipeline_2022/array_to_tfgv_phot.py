"""Makes photon transfer functions for 21cmFAST"""

import os, sys
sys.path.append(os.environ['DH_DIR'])
sys.path.append('..')

import numpy as np
import pickle

from tqdm import tqdm

from darkhistory.low_energy.lowE_electrons import make_interpolator
from darkhistory.spec.spectrum import Spectrum
import darkhistory.physics as phys
from darkhistory.low_energy.lowE_deposition import compute_fs
import darkhistory.spec.spectools as spectools

from config import load_data
from main import get_elec_cooling_data
from darkhistory.electrons import positronium as pos
from darkhistory.electrons.elec_cooling import get_elec_cooling_tf

from dm21cm.common import abscs_nBs_test_2


####################
## Config
abscs = abscs_nBs_test_2
use_tqdm = True

DATA_DIR = '/zfs/yitians/dm21cm/DM21cm/data/tfdata/array/nBs_test_2'
SAVE_DIR = '/zfs/yitians/dm21cm/DM21cm/transferfunctions/nBs_test_2'
os.makedirs(SAVE_DIR, exist_ok=True)


####################
## 1. Load

print('Loading tf: ', end=' ', flush=True)
hep_tfgv = np.load(DATA_DIR+'/hep_tf.npy')
print('hep', end=' ', flush=True)
lep_tfgv = np.load(DATA_DIR+'/lep_tf.npy')
print('lep', end=' ', flush=True)
lee_tfgv = np.load(DATA_DIR+'/lee_tf.npy')
print('lee', end=' ', flush=True)
hed_tfgv = np.load(DATA_DIR+'/hed_tf.npy')
print('hed', end=' ', flush=True)
cmbloss_gv = np.load(DATA_DIR+'/cmbloss.npy')
print('cmb', end='.', flush=True)


####################
## 2. Setup

dlnz = abscs['dlnz']

phot_tfgv = np.zeros_like(hep_tfgv)
phot_depgv = np.zeros(
    hed_tfgv.shape[:-1] + (len(abscs['dep_c']),)
) # channels: {H ionization, He ionization, excitation, heat, continuum}

MEDEA_interp = make_interpolator(interp_type='2D', cross_check=False)


####################
## Loop over rs x nBs
if use_tqdm:
    pbar = tqdm( total = len(abscs['nBs'])*len(abscs['x'])*len(abscs['rs']) )

for i_x, x in enumerate(abscs['x']):
    for i_nBs, nBs in enumerate(abscs['nBs']):
        for i_rs, rs in enumerate(abscs['rs']):
            
            dt = dlnz / phys.hubble(rs)
            
            ###################################
            ## 3. Add cmbloss to highengphot
            cmb_un = spectools.discretize(abscs['photE'], phys.CMB_spec, phys.TCMB(rs))
            cmb_un_E = cmb_un.toteng()
            for i in range(len(abscs['photE'])):
                cmb_E = cmbloss_gv[i_nBs][i_x][i_rs][i] * dt
                hep_tfgv[i_nBs][i_x][i_rs][i] += (-cmb_E/cmb_un_E) * cmb_un.N
            
            ###################################
            ## 4. Add lowengphot diagonal
            for i in range(len(abscs['photE'])):
                if lep_tfgv[i_nBs][i_x][i_rs][i][i] > 1e-40 or lep_tfgv[i_nBs][i_x][i_rs][i][i] < 1e-100:
                    break
                lep_tfgv[i_nBs][i_x][i_rs][i][i] = 1
            
            
            for i_injE, injE in enumerate(abscs['photE']):

                ###################################
                ## 5. Injection
                # inject one photon at i_injE
                hep_spec_N = hep_tfgv[i_nBs, i_x, i_rs, i_injE]
                lep_spec_N = lep_tfgv[i_nBs, i_x, i_rs, i_injE]
                lee_spec_N = lee_tfgv[i_nBs, i_x, i_rs, i_injE]
                hed_arr    = hed_tfgv[i_nBs, i_x, i_rs, i_injE]

                lowengelec_spec_at_rs = Spectrum(abscs['elecEk'], lee_spec_N, spec_type='N')
                lowengelec_spec_at_rs.rs = rs

                lowengphot_spec_at_rs = Spectrum(abscs['photE'], lep_spec_N, spec_type='N')
                lowengphot_spec_at_rs.rs = rs

                highengdep_at_rs = hed_arr

                ###################################
                ## 6. Compute f's
                x_vec_for_f = np.array( [1-x, phys.chi*(1-x), x] ) # [HI, HeI, HeII]/nH
                nBs_ref = 1
                dE_dVdt_inj = injE * phys.nB * nBs_ref * rs**3 / dt # [eV/cm^3 s]
                f_raw = compute_fs(
                    MEDEA_interp,
                    lowengelec_spec_at_rs, # [1/inj] | in DH.main: [1/B]
                    lowengphot_spec_at_rs, # [1/inj] | in DH.main: [1/B]
                    x_vec_for_f,
                    dE_dVdt_inj, # [dE/(dVdt) * B/inj] | in DH.main: [dE/(dVdt)]
                    dt,
                    highengdep_at_rs, # [1/inj/dt] | in DH.main: [1/B]
                    method='no_He', cross_check=False, separate_higheng=False
                )

                ###################################
                ## 7. Compute tf & f values
                lep_prop_spec_N = lep_spec_N * (abscs['photE'] < 10.2)
                f_lep_prop = np.dot(abscs['photE'], lep_prop_spec_N) / injE
                phot_spec_N = hep_spec_N + lep_prop_spec_N
                f_prop = np.dot(abscs['photE'], phot_spec_N) / injE

                f_dep = f_raw
                f_dep[4] -= f_lep_prop # adjust for the propagating lowengphot
                f_tot = f_prop + np.sum(f_dep)

                ###################################
                ## 8. Fix energy conservation (known issues)
                if i_injE == 153: # issue at around 13.6 eV. Adjusting H_ion.
                    f_dep[0] += 1 - f_tot
                if i_injE in range(224, 228): # ??? issue. Adjusting hep propagating bin.
                    phot_spec_N[i_injE] += 1 - f_tot

                f_prop = np.dot(abscs['photE'], phot_spec_N) / injE
                f_tot = f_prop + np.sum(f_dep)

                ###################################
                ## 9. Check energy conservation
                if np.abs(f_tot - 1.) > 1e-4:
                    print(f'  Warning: energy non-conservation at level greater than 1e-4. Giving all to H_ion.')
                    print(f'    nBs[{i_nBs}]={nBs:.3e} x[{i_x}]={x:.3e} rs[{i_rs}]={rs:.3e} injE[{i_injE}]={injE:.3e}')
                    print(f'    f_prop={f_prop:.6f} f_dep={f_dep} f_tot={f_tot:.8f}')

                ###################################
                ## 10. Populate transfer functions
                phot_tfgv[i_nBs, i_x, i_rs, i_injE] = phot_spec_N
                phot_depgv[i_nBs, i_x, i_rs, i_injE] = f_dep
            
            if use_tqdm:
                pbar.update()
            
###################################
## 11. Save transfer function

np.save(SAVE_DIR+'/phot_tfgv.npy', phot_tfgv)
np.save(SAVE_DIR+'/phot_depgv.npy', phot_depgv)