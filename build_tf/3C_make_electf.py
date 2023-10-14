import os
import sys
import time
from tqdm import tqdm
import argparse
import pickle

import numpy as np

from low_energy.lowE_electrons import make_interpolator
from low_energy.lowE_deposition import compute_fs

WDIR = os.environ['DM21CM_DIR']
sys.path.append(WDIR)
from dm21cm.utils import load_h5_dict
import dm21cm.physics as phys

sys.path.append(os.environ['DH_DIR']) # use branch test-dm21cm
from   darkhistory.spec.spectrum import Spectrum
import darkhistory.physics as dh_phys
import darkhistory.spec.spectools as spectools
from   darkhistory.main import get_elec_cooling_data
from   darkhistory.electrons.elec_cooling import get_elec_cooling_tf
from   darkhistory.electrons import positronium as pos
from   darkhistory.config import load_data as dh_load_data


if __name__ == '__main__':

    #===== Config =====
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', type=str, help='run name')
    parser.add_argument('-i', '--i_nBs', type=int, help='nBs index')
    args = parser.parse_args()
    
    run_name = args.name
    data_dir = f"{os.environ['DM21CM_DATA_DIR']}/tf/{run_name}/nBsdhtf"
    save_dir = f"{os.environ['DM21CM_DATA_DIR']}/tf/{run_name}/elec"

    do_not_track_lowengphot = True
    include_f_xray = True
    xray_eng_range = (1e2, 1e4) # [eV]
    use_tqdm = True
    verbose = 0 # {0, 1}
    stop_after_n = np.inf

    #===== Initialize =====
    abscs = load_h5_dict(f"{WDIR}/data/abscissas/abscs_{run_name}.h5")
    dlnz = abscs['dlnz']
    inj_abscs = abscs['elecEk'] + dh_phys.me
    i_xray_fm = np.searchsorted(abscs['photE'], xray_eng_range[0])
    i_xray_to = np.searchsorted(abscs['photE'], xray_eng_range[1])

    MEDEA_interp = make_interpolator(prefix=f'{WDIR}/data/MEDEA')
    coll_ion_sec_elec_specs, coll_exc_sec_elec_specs, ics_engloss_data = get_elec_cooling_data(abscs['elecEk'], abscs['photE'])

    tfgv = np.zeros(( # rxeo. in: elec, out: phot
        len(abscs['rs']),
        len(abscs['x']),
        len(abscs['elecEk']),
        len(abscs['photE'])
    ))
    depgv = np.zeros(( # rxeo. in: elec, out: dep_c
        len(abscs['rs']),
        len(abscs['x']),
        len(abscs['elecEk']),
        len(abscs['dep_c'])
    )) # channels: {H ionization, He ionization, excitation, heat, continuum, xray}

    #===== Load ICS transfer functions =====
    ics_tf_data = dh_load_data('ics_tf', verbose=1)
    ics_thomson_ref_tf = ics_tf_data['thomson']
    ics_rel_ref_tf     = ics_tf_data['rel']
    engloss_ref_tf     = ics_tf_data['engloss']

    #===== Loop =====
    n_run = -1

    if use_tqdm:
        pbar = tqdm( total = len(abscs['rs'])*len(abscs['x']) )

    i_nBs = args.i_nBs
    nBs = abscs['nBs'][i_nBs]

    #===== load nBs dependent photon tf =====
    highengphot_tf_interp = pickle.load(open(f"{data_dir}/highengphot_dhtf_nBs{i_nBs}.p", 'rb'))
    lowengphot_tf_interp  = pickle.load(open(f"{data_dir}/lowengphot_dhtf_nBs{i_nBs}.p", 'rb'))
    lowengelec_tf_interp  = pickle.load(open(f"{data_dir}/lowengelec_dhtf_nBs{i_nBs}.p", 'rb'))
    highengdep_interp     = pickle.load(open(f"{data_dir}/highengdep_dhtf_nBs{i_nBs}.p", 'rb'))

    for i_rs, rs in enumerate(abscs['rs']):
        
        dlnz = abscs['dlnz']
        zplusone_factor = np.exp(dlnz)
        dt = phys.dt_step(rs-1, zplusone_factor) # 21cmFAST dt
        # dt = dlnz / dh_phys.hubble(rs) # DH dt
        # dt = dts[i_rs, 1] # (rs, step) # IDL dt
        
        for i_x, x in enumerate(abscs['x']):

            #===== Get electron cooling tfs =====
            xHII_elec_cooling  = x
            xHeII_elec_cooling = x * phys.chi
            (
                ics_sec_phot_tf, elec_processes_lowengelec_tf,
                deposited_ion_arr, deposited_exc_arr, deposited_heat_arr,
                continuum_loss, deposited_ICS_arr
            ) = get_elec_cooling_tf(
                abscs['elecEk'], abscs['photE'], rs,
                xHII_elec_cooling, xHeII=xHeII_elec_cooling,
                raw_thomson_tf=ics_thomson_ref_tf, 
                raw_rel_tf=ics_rel_ref_tf, 
                raw_engloss_tf=engloss_ref_tf,
                coll_ion_sec_elec_specs=coll_ion_sec_elec_specs, 
                coll_exc_sec_elec_specs=coll_exc_sec_elec_specs,
                ics_engloss_data=ics_engloss_data,
                nBscale=nBs,
            )

            #===== Get photon tfs =====
            xHII = x
            xHeII = x * phys.chi
            highengphot_tf = highengphot_tf_interp.get_tf(xHII, xHeII, rs)
            lowengphot_tf  = lowengphot_tf_interp.get_tf(xHII, xHeII, rs)
            lowengelec_tf  = lowengelec_tf_interp.get_tf(xHII, xHeII, rs)
            highengdep_arr = highengdep_interp.get_val(xHII, xHeII, rs)
            
            for i_injE, injE in enumerate(inj_abscs):
                
                assert n_run <= stop_after_n
                n_run += 1

                #===== Electron injection =====
                # inject one electron (1/2 electron 1/2 positron) at i_injE
                timer = time.time()
                in_spec_elec = Spectrum(abscs['elecEk'], np.zeros_like(abscs['elecEk']), spec_type='N', rs=rs)
                in_spec_elec.N[i_injE] = 1. # [N/inj]

                # storage for electron processes
                highengphot_spec_at_rs = Spectrum(abscs['photE'], np.zeros_like(abscs['photE']), spec_type='N', rs=rs)
                lowengphot_spec_at_rs  = Spectrum(abscs['photE'], np.zeros_like(abscs['photE']), spec_type='N', rs=rs)
                lowengelec_spec_at_rs  = Spectrum(abscs['elecEk'], np.zeros_like(abscs['elecEk']), spec_type='N', rs=rs)
                highengdep_at_rs = np.zeros((4,))

                # Apply the transfer function to the input electron spectrum. 
                # Low energy electrons from electron cooling, per injection event.
                elec_processes_lowengelec_spec = elec_processes_lowengelec_tf.sum_specs(in_spec_elec)

                # norm_fac = inj_rate (dinj/dVdt) * dt / (dBavg/dV) = d(inj in step)/dBavg
                # Add this to lowengelec_at_rs. (Remove norm_fac [N/Bavg in step] -> [N/inj])
                lowengelec_spec_at_rs += elec_processes_lowengelec_spec # * norm_fac

                # Depositions
                deposited_ion  = np.dot(deposited_ion_arr,  in_spec_elec.N) # * norm_fac
                deposited_exc  = np.dot(deposited_exc_arr,  in_spec_elec.N) # * norm_fac
                deposited_heat = np.dot(deposited_heat_arr, in_spec_elec.N) # * norm_fac
                deposited_ICS  = np.dot(deposited_ICS_arr, in_spec_elec.N) # * norm_fac # HED numerical error.
                # units for above quantities: [eV/inj]
                highengdep_at_rs += np.array([
                    deposited_ion/dt,
                    deposited_exc/dt,
                    deposited_heat/dt,
                    deposited_ICS/dt
                ]) # [eV/(inj s)]

                # ICS secondary photon spectrum after electron cooling, 
                # per injection event.
                ics_phot_spec = ics_sec_phot_tf.sum_specs(in_spec_elec) # units see below

                # Get the spectrum from positron annihilation, per injection event.
                # Only half of in_spec_elec is positrons!
                positronium_phot_spec = pos.weighted_photon_spec(abscs['photE']) * (in_spec_elec.totN()/2)
                positronium_phot_spec.switch_spec_type('N')  # units see below

                # Add injected photons + photons from injected electrons
                # to the photon spectrum that got propagated forward. 
                highengphot_spec_at_rs += (ics_phot_spec + positronium_phot_spec) # * norm_fac
                highengphot_spec_at_rs.rs = rs
                # [N/inj]


                #===== Apply photon transfer functions =====
                in_hep_spec = highengphot_spec_at_rs * 1 # copy
                if np.allclose(in_hep_spec.eng, highengphot_tf.eng):
                    in_hep_spec.eng = highengphot_tf.eng # phot
                if np.allclose(highengphot_spec_at_rs.eng, highengphot_tf.eng):
                    highengphot_spec_at_rs.eng = highengphot_tf.eng # phot
                if np.allclose(lowengphot_spec_at_rs.eng, highengphot_tf.eng):
                    lowengphot_spec_at_rs.eng = highengphot_tf.eng # phot
                if np.allclose(lowengelec_spec_at_rs.eng, lowengelec_tf.eng):
                    lowengelec_spec_at_rs.eng = lowengelec_tf.eng # elec
                highengphot_spec_at_rs = highengphot_tf.sum_specs( in_hep_spec )
                lowengphot_spec_at_rs  = lowengphot_tf.sum_specs ( in_hep_spec )
                lowengelec_spec_at_rs += lowengelec_tf.sum_specs ( in_hep_spec )
                highengdep_at_rs += np.dot(np.swapaxes(highengdep_arr, 0, 1), in_hep_spec.N)
                highengphot_spec_at_rs.rs = rs # manually set rs
                lowengphot_spec_at_rs.rs  = rs
                lowengelec_spec_at_rs.rs  = rs

                hep_spec_N = highengphot_spec_at_rs.N
                lep_spec_N = lowengphot_spec_at_rs.N
                

                #===== Compute f's ===== (same as make_phottf)
                x_vec_for_f = np.array( [1-x, dh_phys.chi*(1-x), dh_phys.chi*x] ) # [HI, HeI, HeII]/nH
                nBs_ref = 1
                dE_dVdt_inj = injE * dh_phys.nB * nBs_ref * rs**3 / dt # [eV/cm^3 s]
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

                #===== Compute tf & f values =====
                f_dep = f_raw
                if do_not_track_lowengphot:
                    phot_spec_N = hep_spec_N
                    f_prop = np.dot(abscs['photE'], phot_spec_N) / injE
                else:
                    i_exc_bin = np.searchsorted(spectools.get_bin_bound(abscs['photE']), 10.2) - 1 # 149
                    lep_prop_spec_N = lep_spec_N.copy()
                    lep_prop_spec_N[:i_exc_bin] *= 0.
                    f_lep_prop = np.dot(abscs['photE'], lep_prop_spec_N) / injE
                    phot_spec_N = hep_spec_N + lep_prop_spec_N
                    f_dep[4] -= f_lep_prop # adjust for the propagating lowengphot

                f_prop = np.dot(abscs['photE'], phot_spec_N) / injE
                f_tot = f_prop + np.sum(f_dep)

                #===== Energy conservation =====
                f_prop = np.dot(abscs['photE'], phot_spec_N) / injE
                f_tot = f_prop + np.sum(f_dep)

                f_dep_str = ' '.join([f'{v:.3e}' for v in f_dep])
                print_str = f'{n_run} | {i_rs} {i_x} {i_nBs} {i_injE} | f_prop={f_prop:.6f} f_dep={f_dep_str} f_tot={f_tot:.6f}'
                energy_conservation_threshold = 1e-2
                if np.abs(f_tot - 1.) > energy_conservation_threshold:
                    print_str += f' | Energy error > {energy_conservation_threshold}'
                if verbose >= 1 or np.abs(f_tot - 1.) > energy_conservation_threshold:
                    print(print_str, flush=True)
                
                # enforce conservation by scaling phot_spec_N
                phot_E_now = np.dot(abscs['photE'], phot_spec_N)
                phot_E_target = phot_E_now + injE * (1 - f_tot)
                if phot_E_now == 0:
                    raise ValueError('phot_E_now == 0')
                phot_spec_N[i_injE] *= phot_E_target / phot_E_now
                
                #===== Dependent variables (Xray) =====
                if include_f_xray:
                    f_xray = np.dot(abscs['photE'][i_xray_fm:i_xray_to], phot_spec_N[i_xray_fm:i_xray_to]) / injE
                    if i_xray_fm <= i_injE and i_injE < i_xray_to:
                        f_xray -= phot_spec_N[i_injE] # ignore diagonal for now # NEED TO EXTRACT PROP
                    f_dep = np.append(f_dep, f_xray)

                #===== Populate transfer functions =====
                tfgv[i_rs, i_x, i_injE] = phot_spec_N
                depgv[i_rs, i_x, i_injE] = f_dep
            
            if use_tqdm:
                pbar.update()

    #===== Save =====
    np.save(f'{save_dir}/elec_tfgv_nBs{i_nBs}_rxeo.npy', tfgv)
    np.save(f'{save_dir}/elec_depgv_nBs{i_nBs}_rxeo.npy', depgv)