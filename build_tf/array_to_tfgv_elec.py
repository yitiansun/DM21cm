"""Makes electron transfer functions for 21cmFAST"""

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
shape_convention = 'rexo'

DATA_DIR = '/zfs/yitians/dm21cm/DM21cm/data/tfdata/array/nBs_test_2'
SAVE_DIR = '/zfs/yitians/dm21cm/DM21cm/transferfunctions/nBs_test_2'
os.makedirs(SAVE_DIR, exist_ok=True)


####################
## 1. Setup

dlnz = abscs['dlnz']

elec_phot_tfgv = np.zeros( (
    len(abscs['rs']),
    len(abscs['elecEk']),
    len(abscs['x']),
    len(abscs['photE']),
) )

elec_depgv = np.zeros( (
    len(abscs['rs']),
    len(abscs['elecEk']),
    len(abscs['x']),
    len(abscs['dep_c']),
) ) # channels: {H ionization, He ionization, excitation, heat, continuum}

print(f'shape convention : {shape_convention}')
print(f'elec_phot_tfgv.shape = {elec_phot_tfgv.shape}')
print(f'elec_depgv.shape = {elec_depgv.shape}')


####################
## 2. Load

## load darkhistory transfer functions
ics_tf_data = load_data('ics_tf', verbose=1)
ics_thomson_ref_tf = ics_tf_data['thomson']
ics_rel_ref_tf     = ics_tf_data['rel']
engloss_ref_tf     = ics_tf_data['engloss']

(
    coll_ion_sec_elec_specs, coll_exc_sec_elec_specs,
    ics_engloss_data
) = get_elec_cooling_data(abscs['elecEk'], abscs['photE'])

MEDEA_interp = make_interpolator(interp_type='2D', cross_check=False)


####################
## Loop over rs x
if use_tqdm:
    pbar = tqdm( total = len(abscs['x'])*len(abscs['rs']) )

for i_rs, rs in enumerate(abscs['rs']):
    for i_x, x in enumerate(abscs['x']):
        
        ####################
        ## get tfs for this (rs, x)
        (
            ics_sec_phot_tf, elec_processes_lowengelec_tf,
            deposited_ion_arr, deposited_exc_arr, deposited_heat_arr,
            continuum_loss, deposited_ICS_arr
        ) = get_elec_cooling_tf(
            abscs['elecEk'], abscs['photE'], rs,
            x, xHeII=x,
            raw_thomson_tf=ics_thomson_ref_tf,
            raw_rel_tf=ics_rel_ref_tf,
            raw_engloss_tf=engloss_ref_tf,
            coll_ion_sec_elec_specs=coll_ion_sec_elec_specs,
            coll_exc_sec_elec_specs=coll_exc_sec_elec_specs,
            ics_engloss_data=ics_engloss_data
        )
        
        dt = dlnz / phys.hubble(rs)
        
        
        for i_injE, injE in enumerate(abscs['elecEk']):

            ###################################
            ## Injection one electron
            in_spec_elec_N = np.zeros_like(abscs['elecEk'], dtype=np.float32)
            in_spec_elec_N[i_injE] = 1.

            in_spec_elec = Spectrum(abscs['elecEk'], in_spec_elec_N, spec_type='N') # per injection
            in_spec_elec.rs = rs

            ###################################
            ## Apply elec to elec tf
            # Low energy electrons from electron cooling, per injection event.
            lowengelec_spec_at_rs = elec_processes_lowengelec_tf.sum_specs(in_spec_elec)

            ###################################
            ## Apply deposition tfs
            # All of below value is [eV] per injection event. removed norm_fac.
            deposited_ion = np.dot(deposited_ion_arr, in_spec_elec.N)
            deposited_exc = np.dot(deposited_exc_arr, in_spec_elec.N)
            deposited_heat = np.dot(deposited_heat_arr, in_spec_elec.N)
            deposited_ICS = np.dot(deposited_ICS_arr, in_spec_elec.N) # ??? High-energy deposition numerical error ???
            highengdep_at_rs = np.array([
                deposited_ion/dt,
                deposited_exc/dt,
                deposited_heat/dt,
                deposited_ICS/dt # continuum
            ])

            ###################################
            ## Apply elec to phot tfs

            # ICS secondary photon spectrum after electron cooling, 
            ics_phot_spec = ics_sec_phot_tf.sum_specs(in_spec_elec) # per injection

            # Get the spectrum from positron annihilation, per injection event.
            # Only half of in_spec_elec is positrons!
            positronium_phot_spec = pos.weighted_photon_spec(abscs['photE']) * (
                in_spec_elec.totN()/2
            )
            positronium_phot_spec.switch_spec_type('N')

            # Add injected photons + photons from injected electrons
            # to the photon spectrum that got propagated forward. 
            highengphot_spec_at_rs = ics_phot_spec + positronium_phot_spec
            highengphot_spec_at_rs.rs = rs

            # make an empty lowengphot
            lowengphot_spec_at_rs = highengphot_spec_at_rs * 0
            lowengphot_spec_at_rs.rs = rs

            ###################################
            ## Compute f's for lowengelec
            x_vec_for_f = np.array( [1-x, phys.chi*(1-x), x] ) # [HI, HeI, HeII]/nH
            nBs_ref = 1
            dE_dVdt_inj = (injE + phys.me) * phys.nB * nBs_ref * rs**3 / dt # [eV cm^-3 s^-1]
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
            ## Compute tf & f values
            f_prop = highengphot_spec_at_rs.toteng() / (injE + phys.me)

            f_dep = f_raw
            f_tot = f_prop + np.sum(f_dep)

            ###################################
            ## Fix energy conservation
            # None

            ###################################
            ## Check energy conservation
            if np.abs(f_tot - 1.) > 1e-4:
                print(f'  Warning: energy non-conservation at level greater than 1e-4.')
                print(f'nBs[{i_nBs}]={nBs:.3e} x[{i_x}]={x:.3e} ' + \
                      f'rs[{i_rs}]={rs:.3e} injE[{i_injE}]={injE:.3e} ' + \
                      f'f_dep=[{f_dep[0]:.6f}, {f_dep[1]:.6f}, {f_dep[2]:.6f}, {f_dep[3]:.6f}, {f_dep[4]:.6f}] ' + \
                      f'f_prop={f_prop:.6f} f_tot={f_tot:.8f}')

            ###################################
            ## Populate transfer functions
            elec_phot_tfgv[i_rs, i_injE, i_x] = highengphot_spec_at_rs.N
            elec_depgv[i_rs, i_injE, i_x] = f_dep
                    
        if use_tqdm:
            pbar.update()
            
###################################
## Save transfer function

np.save(SAVE_DIR+'/elec_phot_tfgv_rexo.npy', elec_phot_tfgv)
np.save(SAVE_DIR+'/elec_depgv_rexo.npy', elec_depgv)