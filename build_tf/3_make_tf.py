"""Makes photon transfer functions for 21cmFAST"""

import os, sys
sys.path.append(os.environ['DH_DIR'])
sys.path.append('..')

import numpy as np
import pickle
import time
from tqdm import tqdm

from low_energy.lowE_electrons import make_interpolator
from low_energy.lowE_deposition import compute_fs

from config import load_data
from main import get_elec_cooling_data
from   darkhistory.spec.spectrum import Spectrum
import darkhistory.physics as phys
import darkhistory.spec.spectools as spectools
from   darkhistory.electrons import positronium as pos
from   darkhistory.electrons.elec_cooling import get_elec_cooling_tf


#====================
# 0. Config
run_name = '230408'
tf_type = 'elec'
use_tqdm = True

abscs = pickle.load(open(f'../data/abscissas/abscs_{run_name}.p', 'rb'))
MEDEA_DIR = '../data/MEDEA'
DATA_DIR = f'../data/tf/{run_name}'
SAVE_DIR = f'../data/tf/{run_name}'
os.makedirs(SAVE_DIR, exist_ok=True)


#====================
# 1. Load

print('Loading tf: ', end=' ', flush=True)
hep_tfgv = np.load(DATA_DIR+'/hep_tf_rxneo.npy')
print('hep', end=' ', flush=True)
lep_tfgv = np.load(DATA_DIR+'/lep_tf_rxneo.npy')
print('lep', end=' ', flush=True)
lee_tfgv = np.load(DATA_DIR+'/lee_tf_rxneo.npy')
print('lee', end=' ', flush=True)
hed_tfgv = np.load(DATA_DIR+'/hed_tf_rxneo.npy')
print('hed', end=' ', flush=True)
cmbloss_gv = np.load(DATA_DIR+'/cmbloss_rxneo.npy')
print('cmb', end='.', flush=True)


#====================
# 2. Setup

dlnz = abscs['dlnz']

phot_tfgv = np.zeros_like(hep_tfgv)
phot_depgv = np.zeros(
    hed_tfgv.shape[:-1] + (len(abscs['dep_c']),)
) # channels: {H ionization, He ionization, excitation, heat, continuum}

MEDEA_interp = make_interpolator(prefix=MEDEA_DIR)


#====================
# Loop over rs x nBs
if use_tqdm:
    pbar = tqdm( total = len(abscs['rs'])*len(abscs['x'])*len(abscs['nBs']) )

for i_rs, rs in enumerate(abscs['rs']):
    
    dt = dlnz / phys.hubble(rs)
            
    #==============================
    # 3. Add cmbloss to highengphot
    cmb_un = spectools.discretize(abscs['photE'], phys.CMB_spec, phys.TCMB(rs))
    cmb_un_E = cmb_un.toteng()
    
    for i_x, x in enumerate(abscs['x']):
        for i_nBs, nBs in enumerate(abscs['nBs']):
            
            for i in range(len(abscs['photE'])):
                cmb_E = cmbloss_gv[i_rs][i_x][i_nBs][i] * dt
                hep_tfgv[i_rs][i_x][i_nBs][i] += (-cmb_E/cmb_un_E) * cmb_un.N

            #==============================
            # 4. Add lowengphot diagonal
            if nBs == 0: # lowengphot is 0 when nBs is 0
                raise NotImplementedError
            for i in range(len(abscs['photE'])):
                if lep_tfgv[i_rs][i_x][i_nBs][i][i] > 1e-40:
                    break
                lep_tfgv[i_rs][i_x][i_nBs][i][i] = 1
            
            for i_injE, injE in enumerate(abscs['photE']):

                #==============================
                # 5. Injection
                # inject one photon at i_injE
                timer = time.time()
                
                hep_spec_N = hep_tfgv[i_rs, i_x, i_nBs, i_injE]
                lep_spec_N = lep_tfgv[i_rs, i_x, i_nBs, i_injE]
                lee_spec_N = lee_tfgv[i_rs, i_x, i_nBs, i_injE]
                hed_arr    = hed_tfgv[i_rs, i_x, i_nBs, i_injE]

                lowengelec_spec_at_rs = Spectrum(abscs['elecEk'], lee_spec_N, spec_type='N')
                lowengelec_spec_at_rs.rs = rs

                lowengphot_spec_at_rs = Spectrum(abscs['photE'], lep_spec_N, spec_type='N')
                lowengphot_spec_at_rs.rs = rs

                highengdep_at_rs = hed_arr

                #==============================
                # 6. Compute f's
                x_vec_for_f = np.array( [1-x, phys.chi*(1-x), x] ) # [HI, HeI, HeII]/nH
                nBs_ref = 1
                dE_dVdt_inj = injE * phys.nB * nBs_ref * rs**3 / dt # [eV/cm^3 s]
                # in DH.main: (dN_inj/dB) / (dE_inj  /dVdt)
                # here:       (dN_inj   ) / (dE_injdB/dVdt)
                f_low, f_high = compute_fs(
                    MEDEA_interp=MEDEA_interp,
                    rs=rs,
                    x=x_vec_for_f,
                    elec_spec=lowengelec_spec_at_rs,
                    phot_spec=lowengphot_spec_at_rs,
                    dE_dVdt_inj=dE_dVdt_inj,
                    dt=dt,
                    highengdep=highengdep_at_rs,
                    cmbloss=0, # turned off in darkhistory main as well
                    method='no_He',
                    cross_check=False,
                    ion_old=False
                )
                f_raw = f_low + f_high

                #==============================
                # 7. Compute tf & f values
                lep_prop_spec_N = lep_spec_N * (abscs['photE'] < 10.2)
                f_lep_prop = np.dot(abscs['photE'], lep_prop_spec_N) / injE
                phot_spec_N = hep_spec_N + lep_prop_spec_N
                f_prop = np.dot(abscs['photE'], phot_spec_N) / injE

                f_dep = f_raw
                f_dep[4] -= f_lep_prop # adjust for the propagating lowengphot
                f_tot = f_prop + np.sum(f_dep)

                #==============================
                # 8. Fix energy conservation (known issues)
                if i_injE == 153: # issue at around 13.6 eV. Adjusting H_ion.
                    f_dep[0] += 1 - f_tot
                if i_injE in range(224, 228): # ??? issue. Adjusting hep propagating bin.
                    phot_spec_N[i_injE] += 1 - f_tot

                f_prop = np.dot(abscs['photE'], phot_spec_N) / injE
                f_tot = f_prop + np.sum(f_dep)

                #==============================
                # 9. Check energy conservation
                if np.abs(f_tot - 1.) > 1e-2:
                    print(f'  Warning: energy non-conservation at level greater than 1e-2. Giving all to propagating photon.')
                    print(f'    nBs[{i_nBs}]={nBs:.3e} x[{i_x}]={x:.3e} rs[{i_rs}]={rs:.3e} injE[{i_injE}]={injE:.3e}')
                    print(f'    f_prop={f_prop:.6f} f_dep={f_dep} f_tot={f_tot:.8f}')
                
                #f_dep[0] += 1 - f_tot
                phot_spec_N[i_injE] += 1 - f_tot

                #==============================
                # 10. Populate transfer functions
                phot_tfgv[i_rs, i_x, i_nBs, i_injE] = phot_spec_N
                phot_depgv[i_rs, i_x, i_nBs, i_injE] = f_dep
            
            if use_tqdm:
                pbar.update()
            
#==============================
# 11. Save transfer function

np.save(SAVE_DIR+'/phot_tfgv.npy', phot_tfgv)
np.save(SAVE_DIR+'/phot_depgv.npy', phot_depgv)