"""Main evolution function."""

import os
import sys
import time
import logging

import numpy as np
from astropy.cosmology import Planck18

import py21cmfast as p21c
from py21cmfast import cache_tools

sys.path.append(os.environ['DH_DIR']) # use branch test_dm21cm
from darkhistory.spec.spectrum import Spectrum

sys.path.append("..")
import dm21cm.physics as phys
from dm21cm.dm_params import DMParams
from dm21cm.dh_wrapper import DarkHistoryWrapper, TransferFunctionWrapper
from dm21cm.utils import split_xray, get_z_edges, gen_injection_boxes, p21_step, load_dict
from dm21cm.data_cacher import Cacher

logging.getLogger().setLevel(logging.INFO)
logging.getLogger('21cmFAST').setLevel(logging.CRITICAL+1)
logging.getLogger('py21cmfast._utils').setLevel(logging.CRITICAL+1)
logging.getLogger('py21cmfast.wrapper').setLevel(logging.CRITICAL+1)


def evolve(run_name, z_start=..., z_end=..., zplusone_step_factor=...,
           dm_params=..., enable_elec=False, tf_version=...,
           p21c_initial_conditions=...,
           rerun_DH=False, clear_cache=False,
           use_tqdm=True, debug=False,
           debug_uniform_xray=False,):
    """
    Main evolution function.

    Args:
        run_name (str): Name of run. Used for cache directory.
        z_start (float): Starting redshift.
        z_end (float): Ending redshift.
        zplusone_step_factor (float): (1+z) / (1+z_one_step_earlier). Must be greater than 1.
        dm_params (DMParams): Dark matter (DM) parameters.
        enable_elec (bool): Whether to enable electron injection.
        tf_version (str): Version of DarkHistory transfer function to use.
        p21c_initial_conditions (InitialConditions): Initial conditions from py21cmfast.
        
        rerun_DH (bool): Whether to rerun DarkHistory to get initial values.
        clear_cache (bool): Whether to clear cache for 21cmFAST.
        force_reload_tf (bool): Whether to force reload transfer functions. Use when changing dhtf_version.
        use_tqdm (bool): Whether to use tqdm progress bars.

        debug (bool) : Whether to turn on debug mode.
        debug_uniform_xray (bool) : Whether to use a uniform xray spectrum.
        
    Returns:
        debug_dict if debug is set. Otherwise None.
    """

    logging.info(f'Using 21cmFAST version {p21c.__version__}')

    #===== cache =====
    p21c.config['direc'] = f"{os.environ['P21C_CACHE_DIR']}/{run_name}"
    logging.info(f"Cache dir: {p21c.config['direc']}")
    os.makedirs(p21c.config['direc'], exist_ok=True)
    if clear_cache:
        cache_tools.clear_cache()

    #===== initialize =====
    #--- physics parameters ---
    p21c.global_params.Z_HEAT_MAX = z_start
    p21c.global_params.ZPRIME_STEP_FACTOR = zplusone_step_factor
    p21c.global_params.CLUMPING_FACTOR = 1.

    abscs = load_dict(f'../data/abscissas/abscs_{tf_version}.h5')
    if not np.isclose(np.log(zplusone_step_factor), abscs['dlnz']):
        raise ValueError('zplusone_step_factor and dhtf_version mismatch')
    dm_params.set_inj_specs(abscs)
    
    box_dim = p21c_initial_conditions.user_params.HII_DIM
    box_len = p21c_initial_conditions.user_params.BOX_LEN
    cosmo = Planck18

    #--- DarkHistory and transfer functions ---
    dh_wrapper = DarkHistoryWrapper(
        dm_params,
        prefix=p21c.config[f'direc'],
    )
    tf_prefix = f"{os.environ['DM21CM_DATA_DIR']}/tf/{tf_version}"
    tf_wrapper = TransferFunctionWrapper(
        box_dim = box_dim,
        abscs = abscs,
        tf_prefix = tf_prefix,
        enable_elec = enable_elec,
    )

    #--- xray ---
    ex_lo, ex_hi = 1e2, 1e4 # [eV]
    ix_lo = np.searchsorted(abscs['photE'], ex_lo) # i of first bin greater than ex_lo, excluded
    ix_hi = np.searchsorted(abscs['photE'], ex_hi) # i of first bin greater than ex_hi, included

    xray_fn = f"{p21c.config['direc']}/xray_brightness.h5"
    if os.path.isfile(xray_fn):
        os.remove(xray_fn)
    xray_cacher = Cacher(data_path=xray_fn, cosmo=cosmo, N=box_dim, dx=box_len/box_dim)

    #--- redshift stepping ---
    z_edges = get_z_edges(z_start, z_end, 1.01)
    def get_time_step(i_z):
        z_current = z_edges[i_z]
        z_next = z_edges[i_z+1]
        dt = ( cosmo.age(z_next) - cosmo.age(z_current) ).to('s').value # [s]
        return z_current, z_next, dt

    #===== initial step =====
    perturbed_field = p21c.perturb_field(redshift=z_edges[0], init_boxes=p21c_initial_conditions)
    spin_temp, ionized_box, brightness_temp = p21_step(z_edges[0], perturbed_field, None, None)

    dh_wrapper.evolve(end_rs=(1+z_start)*0.9, rerun=rerun_DH)
    dh_wrapper.match(spin_temp, ionized_box)
    phot_bath_spec = dh_wrapper.get_phot_bath(rs=1+z_edges[0])

    #===== main loop =====
    i_xray_loop_start = 0 # where we start looking for annuli
    records = []

    z_iterator = range(len(z_edges)-1)
    if use_tqdm:
        from tqdm import tqdm
        z_iterator = tqdm(z_iterator)

    for i_z in z_iterator:

        z_current, z_next, dt = get_time_step(i_z)

        timer_start = time.time()
        print(f'step {i_z} z: {z_current:.3f}->{z_next:.3f} ', end='', flush=True)
        
        nBavg = phys.n_B * (1+z_current)**3 # [Bavg / (physical cm)^3]
        delta_plus_one_box = 1 + np.asarray(perturbed_field.density)
        rho_DM_box = delta_plus_one_box * phys.rho_DM * (1+z_current)**3 # [eV/(physical cm)^3]
        x_e_box = np.asarray(1 - ionized_box.xH_box) # check this
        inj_per_Bavg_box = phys.inj_rate(rho_DM_box, dm_params) * dt * dm_params.struct_boost(1+z_current) / nBavg # [inj/Bavg]
        
        tf_wrapper.init_step(
            rs = 1 + z_current,
            delta_plus_one_box = delta_plus_one_box,
            x_e_box = x_e_box,
        )
        
        #===== photon injection and energy deposition =====
        #--- xray ---
        for i_z_shell in range(i_xray_loop_start, i_z):

            xray_brightness_box, xray_spec, is_box_average = xray_cacher.get_annulus_data(
                z_current, z_edges[i_z_shell], z_edges[i_z_shell+1]
            )
            # If we are smoothing on the scale of the box then dump to the global bath spectrum.
            # The deposition will happen later, and we will not revisit this shell.
            if is_box_average:
                phot_bath_spec.N += xray_brightness_box[0, 0, 0] * xray_spec.N
                i_xray_loop_start = max(i_z_shell+1, i_xray_loop_start)
                continue

            tf_wrapper.inject_phot(xray_spec, inject_type='xray', weight_box=xray_brightness_box)

        print(f'xray: {time.time()-timer_start:.3f} ', end='', flush=True)
        timer_start = time.time()

        #--- homogeneous bath ---
        tf_wrapper.inject_phot(phot_bath_spec, inject_type='bath')
        
        #--- dark matter (on-the-spot) ---
        tf_wrapper.inject_from_dm(dm_params, inj_per_Bavg_box)

        print(f'bath+dm: {time.time()-timer_start:.3f} ', end='', flush=True)
        timer_start = time.time()
        
        #===== 21cmFAST step =====
        perturbed_field = p21c.perturb_field(redshift=z_next, init_boxes=p21c_initial_conditions)    
        input_heating, input_ionization, input_jalpha = gen_injection_boxes(z_next, p21c_initial_conditions)
        tf_wrapper.populate_injection_boxes(input_heating, input_ionization, input_jalpha)
        
        spin_temp, ionized_box, brightness_temp = p21_step(
            z_next, perturbed_field, spin_temp, ionized_box,
            input_heating, input_ionization, input_jalpha
        )

        print(f'21cmfast: {time.time()-timer_start:.3f} ', end='', flush=True)
        timer_start = time.time()
        
        #===== prepare spectra for next step =====
        attenuation_arr = tf_wrapper.attenuation_arr(rs=1+z_current, x=np.mean(x_e_box))
        xray_cacher.advance_spectrum(attenuation_arr, z_next)

        prop_phot_N, emit_phot_N = tf_wrapper.prop_phot_N, tf_wrapper.emit_phot_N
        emit_bath_N, emit_xray_N = split_xray(emit_phot_N, ix_lo, ix_hi)
        out_phot_N = prop_phot_N + emit_bath_N # photons not emitted to the xray band are added to the bath
        
        # Prepare the bath spectrum for the next step    
        out_phot_spec = Spectrum(abscs['photE'], out_phot_N, rs=1+z_current, spec_type='N')
        out_phot_spec.redshift(1+z_next)
        phot_bath_spec = out_phot_spec
        
        # Redshift the x-ray spectrum to the next timestep. Then cache the brightness box and spectrum
        xray_spec = Spectrum(abscs['photE'], emit_xray_N, rs=1+z_current, spec_type='N') # [photon / Bavg]
        xray_spec.redshift(1+z_next)
        
        xray_e_box = tf_wrapper.xray_E_box / np.dot(abscs['photE'], emit_xray_N) # [1 (relative energy) / Bavg]
        xray_cacher.set_cache(z_current, xray_e_box, xray_spec)
        
        #===== save some global quantities =====
        dE_inj_per_Bavg = dm_params.eng_per_inj * np.mean(inj_per_Bavg_box) # [eV per Bavg]
        dE_inj_per_Bavg_unclustered = dE_inj_per_Bavg / dm_params.struct_boost(1+z_current)
        
        record = {
            'z'   : z_next,
            'T_s' : np.mean(spin_temp.Ts_box), # [mK]
            'T_b' : np.mean(brightness_temp.brightness_temp), # [K]
            'T_k' : np.mean(spin_temp.Tk_box), # [K]
            'x_e' : np.mean(1 - ionized_box.xH_box), # [1]
            'E_phot' : phot_bath_spec.toteng(), # [eV/Bavg]
            'dE_inj_per_B' : dE_inj_per_Bavg,
            'f_ion'  : np.mean(tf_wrapper.dep_box[...,0] + tf_wrapper.dep_box[...,1]) / dE_inj_per_Bavg_unclustered,
            'f_exc'  : np.mean(tf_wrapper.dep_box[...,2]) / dE_inj_per_Bavg_unclustered,
            'f_heat' : np.mean(tf_wrapper.dep_box[...,3]) / dE_inj_per_Bavg_unclustered,
        }
        records.append(record)

        print(f'others: {time.time()-timer_start:.3f}')
        
    arr_records = {k: np.array([r[k] for r in records]) for k in records[0].keys()}
    np.save(f"{os.environ['DM21CM_DIR']}/data/run_info/{run_name}_records", arr_records)

    if debug:
        return {}
    else:
        return None