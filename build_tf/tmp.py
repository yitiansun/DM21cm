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

from dm21cm.common import abscs_nBs_test_2


####################
## Config
abscs = abscs_nBs_test_2
DATA_DIR = '/zfs/yitians/DM21cm/data/tfdata/array/nBs_test_2'
SAVE_DIR = '/zfs/yitians/DM21cm/transferfunctions/nBs_test_2'
os.makedirs(SAVE_DIR, exist_ok=True)









####################
## load
binning = load_data('binning', verbose=verbose)
photeng = binning['phot']
eleceng = binning['elec']

dep_tf_data = load_data('dep_tf', verbose=verbose)
highengphot_tf_interp = dep_tf_data['highengphot']
lowengphot_tf_interp  = dep_tf_data['lowengphot']
lowengelec_tf_interp  = dep_tf_data['lowengelec']
highengdep_interp     = dep_tf_data['highengdep']

ics_tf_data = load_data('ics_tf', verbose=verbose)
ics_thomson_ref_tf = ics_tf_data['thomson']
ics_rel_ref_tf     = ics_tf_data['rel']
engloss_ref_tf     = ics_tf_data['engloss']

####################
## spectrum
in_spec_elec = pppc.get_pppc_spec(mDM, eleceng, primary, 'elec')
in_spec_phot = pppc.get_pppc_spec(mDM, photeng, primary, 'phot')

####################
## before loop
dlnz = 0.001???
dt   = dlnz * coarsen_factor / phys.hubble(rs)

def norm_fac(rs):
    # Normalization to convert from per injection event to 
    # per baryon per dlnz step. 
    return rate_func_N(rs) * (
        dlnz * coarsen_factor / phys.hubble(rs) / (phys.nB * rs**3)
    )


####################
## high eng elec tf
(
    coll_ion_sec_elec_specs, coll_exc_sec_elec_specs,
    ics_engloss_data
) = get_elec_cooling_data(eleceng, photeng)


MEDEA_interp = make_interpolator(interp_type='2D', cross_check=cross_check)

(
    ics_sec_phot_tf, elec_processes_lowengelec_tf,
    deposited_ion_arr, deposited_exc_arr, deposited_heat_arr,
    continuum_loss, deposited_ICS_arr
) = get_elec_cooling_tf(
        eleceng, photeng, rs,
        xHII_elec_cooling, xHeII=xHeII_elec_cooling,
        raw_thomson_tf=ics_thomson_ref_tf, 
        raw_rel_tf=ics_rel_ref_tf, 
        raw_engloss_tf=engloss_ref_tf,
        coll_ion_sec_elec_specs=coll_ion_sec_elec_specs, 
        coll_exc_sec_elec_specs=coll_exc_sec_elec_specs,
        ics_engloss_data=ics_engloss_data
)

# Low energy electrons from electron cooling, per injection event.
elec_processes_lowengelec_spec = (
    elec_processes_lowengelec_tf.sum_specs(in_spec_elec)
)

# Add this to lowengelec_at_rs. 
lowengelec_spec_at_rs += (elec_processes_lowengelec_spec*norm_fac(rs))

deposited_ion = np.dot(deposited_ion_arr, in_spec_elec.N*norm_fac(rs)) # /B
deposited_exc = np.dot(deposited_exc_arr, in_spec_elec.N*norm_fac(rs)) # /B
deposited_heat = np.dot(deposited_heat_arr, in_spec_elec.N*norm_fac(rs)) # /B
deposited_ICS  = np.dot(deposited_ICS_arr, in_spec_elec.N*norm_fac(rs)) # /B

####################
## Photons from Injected Electrons

ics_phot_spec = ics_sec_phot_tf.sum_specs(in_spec_elec) # per injection event

# Get the spectrum from positron annihilation, per injection event.
# Only half of in_spec_elec is positrons!
positronium_phot_spec = pos.weighted_photon_spec(photeng) * (in_spec_elec.totN()/2)
positronium_phot_spec.switch_spec_type('N')

highengphot_spec_at_rs += (in_spec_phot + ics_phot_spec + positronium_phot_spec) * norm_fac(rs)

highengdep_at_rs += np.array([
    deposited_ion/dt,
    deposited_exc/dt,
    deposited_heat/dt,
    deposited_ICS/dt
])

x_vec_for_f = np.array([
        1. - phys.xHII_std(rs), 
        phys.chi - phys.xHeII_std(rs), 
        phys.xHeII_std(rs)
])

f_raw = compute_fs(
    MEDEA_interp, lowengelec_spec_at_rs, lowengphot_spec_at_rs,
    x_vec_for_f, rate_func_eng_unclustered(rs), dt,
    highengdep_at_rs, method=compute_fs_method, cross_check=cross_check
)

# Save the f_c(z) values.
f_low  = np.concatenate((f_low,  [f_raw[0]]))
f_high = np.concatenate((f_high, [f_raw[1]]))

# Save CMB upscattered rate and high-energy deposition rate.
highengdep_grid = np.concatenate(
    (highengdep_grid, [highengdep_at_rs])
)

# Compute f for TLA: sum of low and high. 
f_H_ion = f_raw[0][0] + f_raw[1][0]
f_exc   = f_raw[0][2] + f_raw[1][2]
f_heat  = f_raw[0][3] + f_raw[1][3]

if compute_fs_method == 'old':
    # The old method neglects helium.
    f_He_ion = 0. 
else:
    f_He_ion = f_raw[0][1] + f_raw[1][1]

    
rs_to_interp = np.exp(np.log(rs) - dlnz * coarsen_factor/2)

highengphot_tf, lowengphot_tf, lowengelec_tf, highengdep_arr, prop_tf = (
    get_tf(
        rs, xHII_to_interp, xHeII_to_interp,
        dlnz, coarsen_factor=coarsen_factor
    )
)

# Get the spectra for the next step by applying the 
# transfer functions. 
highengdep_at_rs = np.dot(
    np.swapaxes(highengdep_arr, 0, 1),
    out_highengphot_specs[-1].N
)
highengphot_spec_at_rs = highengphot_tf.sum_specs( out_highengphot_specs[-1] )
lowengphot_spec_at_rs  = lowengphot_tf.sum_specs ( out_highengphot_specs[-1] )
lowengelec_spec_at_rs  = lowengelec_tf.sum_specs ( out_highengphot_specs[-1] )

# Some processing to get the data into presentable shape. 
f_low_dict = {
    'H ion':  f_low[:,0],
    'He ion': f_low[:,1],
    'exc':    f_low[:,2],
    'heat':   f_low[:,3],
    'cont':   f_low[:,4]
}
f_high_dict = {
    'H ion':  f_high[:,0],
    'He ion': f_high[:,1],
    'exc':    f_high[:,2],
    'heat':   f_high[:,3],
    'cont':   f_high[:,4]
}
f = {
    'low': f_low_dict, 'high': f_high_dict
}


def get_elec_cooling_data(eleceng, photeng):
    """
    Returns electron cooling data for use in :func:`main.evolve`.

    Parameters
    ----------
    eleceng : ndarray
        The electron energy abscissa. 
    photeng : ndarray
        The photon energy abscissa. 

    Returns
    -------
    tuple of ndarray
        A tuple with containing 3 tuples. The first tuple contains the 
        normalized collisional ionization scattered electron spectrum for 
        HI, HeI and HeII. The second contains the normalized collisional 
        excitation scattered electron spectrum for HI, HeI and HeII. The 
        last tuple is an 
        :class:`.EnglossRebinData` object for use in rebinning ICS energy loss data to obtain the ICS scattered 
        electron spectrum. 
    """

    # Compute the (normalized) collisional ionization spectra.
    coll_ion_sec_elec_specs = (
        phys.coll_ion_sec_elec_spec(eleceng, eleceng, species='HI'),
        phys.coll_ion_sec_elec_spec(eleceng, eleceng, species='HeI'),
        phys.coll_ion_sec_elec_spec(eleceng, eleceng, species='HeII')
    )
    # Compute the (normalized) collisional excitation spectra.
    id_mat = np.identity(eleceng.size)

    # Electron with energy eleceng produces a spectrum with one particle
    # of energy eleceng - phys.lya.eng. Similar for helium. 
    coll_exc_sec_elec_tf_HI = tf.TransFuncAtRedshift(
        np.squeeze(id_mat[:, np.where(eleceng > phys.lya_eng)]),
        in_eng = eleceng, rs = -1*np.ones_like(eleceng),
        eng = eleceng[eleceng > phys.lya_eng] - phys.lya_eng,
        dlnz = -1, spec_type = 'N'
    )

    coll_exc_sec_elec_tf_HeI = tf.TransFuncAtRedshift(
        np.squeeze(
            id_mat[:, np.where(eleceng > phys.He_exc_eng['23s'])]
        ),
        in_eng = eleceng, rs = -1*np.ones_like(eleceng),
        eng = (
            eleceng[eleceng > phys.He_exc_eng['23s']] 
            - phys.He_exc_eng['23s']
        ), 
        dlnz = -1, spec_type = 'N'
    )

    coll_exc_sec_elec_tf_HeII = tf.TransFuncAtRedshift(
        np.squeeze(id_mat[:, np.where(eleceng > 4*phys.lya_eng)]),
        in_eng = eleceng, rs = -1*np.ones_like(eleceng),
        eng = eleceng[eleceng > 4*phys.lya_eng] - 4*phys.lya_eng,
        dlnz = -1, spec_type = 'N'
    )

    # Rebin the data so that the spectra stored above now have an abscissa
    # of eleceng again (instead of eleceng - phys.lya_eng for HI etc.)
    coll_exc_sec_elec_tf_HI.rebin(eleceng)
    coll_exc_sec_elec_tf_HeI.rebin(eleceng)
    coll_exc_sec_elec_tf_HeII.rebin(eleceng)

    # Put them in a tuple.
    coll_exc_sec_elec_specs = (
        coll_exc_sec_elec_tf_HI.grid_vals,
        coll_exc_sec_elec_tf_HeI.grid_vals,
        coll_exc_sec_elec_tf_HeII.grid_vals
    )

    # Store the ICS rebinning data for speed. Contains information
    # that makes converting an energy loss spectrum to a scattered
    # electron spectrum fast. 
    ics_engloss_data = EnglossRebinData(eleceng, photeng, eleceng)

    return (
        coll_ion_sec_elec_specs, coll_exc_sec_elec_specs, ics_engloss_data
    )