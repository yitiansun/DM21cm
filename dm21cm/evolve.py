"""Main evolution function."""

import os
import sys
import logging
import gc
import pickle
import shutil

import numpy as np
from scipy import interpolate
from astropy.cosmology import Planck18
import astropy.units as u

from jax import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp

import py21cmfast as p21c
from py21cmfast import cache_tools

sys.path.append(os.environ['DH_DIR'])
from darkhistory.spec.spectrum import Spectrum
from darkhistory.history.reionization import alphaA_recomb
from darkhistory.history.tla import compton_cooling_rate

WDIR = os.environ['DM21CM_DIR']
sys.path.append(WDIR)
import dm21cm.physics as phys
from dm21cm.dh_wrappers import DarkHistoryWrapper, TransferFunctionWrapper
from dm21cm.utils import load_h5_dict
from dm21cm.data_cacher import Cacher
#from dm21cm.profiler import Profiler
from dm21cm.spectrum import AttenuatedSpectrum
from dm21cm.interpolators_jax import SFRDInterpolator

logging.getLogger().setLevel(logging.INFO)
logging.getLogger('21cmFAST').setLevel(logging.CRITICAL+1)
logging.getLogger('py21cmfast._utils').setLevel(logging.CRITICAL+1)
logging.getLogger('py21cmfast.wrapper').setLevel(logging.CRITICAL+1)

L_X_numerical_factor = 1e60 # make float happy

def evolve(run_name, z_start=..., z_end=..., zplusone_step_factor=...,
           dm_params=..., enable_elec=False,
           p21c_initial_conditions=...,
           clear_cache=True,
           use_tqdm=True,
           use_21cmfast_xray=False,
           debug_flags=[],
           save_dir=None,
           debug_turn_off_pop2ion=False,
           debug_copy_dh_init=None,
           tf_on_device=True,
           debug_unif_delta_dep=False, # NOTE: currently, just affects the denom, phot_dep still delta dependent
           debug_unif_delta_tf_param=False,
           st_multiplier=1.,
           debug_nodplus1=False,
           debug_xray_Rmax_bath=None,
           debug_xray_Rmax_shell=None,
           debug_xray_Rmax_p21c=None,
           ):
    """
    Main evolution function.

    Args:
        run_name (str): Name of run. Used for cache directory.
        z_start (float): Starting redshift.
        z_end (float): Ending redshift.
        zplusone_step_factor (float): (1+z) / (1+z_one_step_earlier). Must be greater than 1.
        dm_params (DMParams): Dark matter (DM) parameters.
        enable_elec (bool): Whether to enable electron injection.
        p21c_initial_conditions (InitialConditions): Initial conditions from py21cmfast.

        clear_cache (bool): Whether to clear cache for 21cmFAST.
        force_reload_tf (bool): Whether to force reload transfer functions. Use when changing dhtf_version.
        use_tqdm (bool): Whether to use tqdm progress bars.
        debug_flags (list): List of debug flags. Can contain:
            'uniform-xray' : Force xray to inject uniformly.
            xraychecks: (Use x_e instead of 1-x_H for tf is now the baseline.)
            'xraycheck' : Xray check mode. And only one of the following:
                'xc-bath' : Xray check: include bath (larger box injection) as well.
                'xc-01attenuation' : Xray check: approximate attenuation.
                'xc-noatten' : Xray check: no attenuation.
                'xc-custom-SFRD' : Xray check: use custom SFRD.
                'xc-ots: Xray check: use on-the-spot deposition.
        
        DarkHistory checks:
            debug_dhc_delta_one (bool): Whether to use delta = 1.
        
    Returns:
        dict: Dictionary of results.
    """

    logging.info(f'Using 21cmFAST version {p21c.__version__}')

    #===== data and cache =====
    data_dir = os.environ['DM21CM_DATA_DIR']
    p21c.config['direc'] = f"{os.environ['P21C_CACHE_DIR']}/{run_name}"
    logging.info(f"Cache dir: {p21c.config['direc']}")
    os.makedirs(p21c.config['direc'], exist_ok=True)
    if clear_cache:
        cache_tools.clear_cache()
    
    gc.collect()

    #===== initialize =====
    #--- physics parameters ---
    EPSILON = 1e-6
    p21c.global_params.Z_HEAT_MAX = z_start + EPSILON
    p21c.global_params.ZPRIME_STEP_FACTOR = zplusone_step_factor
    p21c.global_params.CLUMPING_FACTOR = 1.
    if debug_turn_off_pop2ion:
        p21c.global_params.Pop2_ion = 0.
    if debug_xray_Rmax_p21c is not None:
        p21c.global_params.R_XLy_MAX = debug_xray_Rmax_p21c
    if use_21cmfast_xray:
        astro_params = p21c.AstroParams(L_X = 40.) # log10 value
    else:
        astro_params = p21c.AstroParams(L_X = 0.) # log10 value

    abscs = load_h5_dict(f"{data_dir}/abscissas.h5")
    if not np.isclose(np.log(zplusone_step_factor), abscs['dlnz']):
        raise ValueError('zplusone_step_factor and abscs mismatch')
    dm_params.set_inj_specs(abscs)

    box_dim = p21c_initial_conditions.user_params.HII_DIM
    box_len = p21c_initial_conditions.user_params.BOX_LEN
    cosmo = Planck18

    #--- DarkHistory and transfer functions ---
    tf_wrapper = TransferFunctionWrapper(
        box_dim = box_dim,
        abscs = abscs,
        prefix = data_dir,
        enable_elec = enable_elec,
        on_device = tf_on_device,
    )

    #--- xraycheck ---
    delta_cacher = Cacher(
        data_path=f"{p21c.config['direc']}/xraycheck_brightness.h5",
        cosmo=cosmo, N=box_dim, dx=box_len/box_dim,
        shell_Rmax=debug_xray_Rmax_shell,
        Rmax=debug_xray_Rmax_bath,
    )
    delta_cacher.clear_cache()

    
    xray_eng_lo = 0.5 * 1000 # [eV]
    xray_eng_hi = 10.0 * 1000 # [eV]
    xray_i_lo = np.searchsorted(abscs['photE'], xray_eng_lo)
    xray_i_hi = np.searchsorted(abscs['photE'], xray_eng_hi)

    # res_dict = np.load(f"{data_dir}/xray_tables.npz", allow_pickle=True)
    # z_range, delta_range, r_range = res_dict['SFRD_Params']

    # cond_sfrd_table = res_dict['Cond_SFRD_Table']
    # st_sfrd_table =  res_dict['ST_SFRD_Table']
    sfrd_tables = load_h5_dict(f"{data_dir}/sfrd_tables.h5")
    z_range = sfrd_tables['z_range']
    delta_range = sfrd_tables['delta_range']
    r_range = sfrd_tables['r_range']
    cond_sfrd_table = sfrd_tables['cond_sfrd_table']
    st_sfrd_table = sfrd_tables['st_sfrd_table']

    # Takes the redshift as `z`
    # The overdensity parameter smoothed on scale `R`
    # The smoothing scale `R` in units of Mpc
    # Returns the conditional PS star formation rate density in [M_Sun / Mpc^3 / s]
    #Cond_SFRD_Interpolator = interpolate.RegularGridInterpolator((z_range, delta_range, r_range), cond_sfrd_table) # scipy bad
    Cond_SFRD_Interpolator = SFRDInterpolator(z_range, delta_range, r_range, cond_sfrd_table) # jax good

    # Takes the redshift as `z`
    # Returns the mean ST star formation rate density star formation rate density in [M_Sun / Mpc^3 / s]
    ST_SFRD_Interpolator = interpolate.interp1d(z_range, st_sfrd_table * st_multiplier)

    #--- redshift stepping ---
    z_edges = get_z_edges(z_start, z_end, p21c.global_params.ZPRIME_STEP_FACTOR)

    #===== initial steps =====
    dh_wrapper = DarkHistoryWrapper(
        dm_params,
        prefix = p21c.config[f'direc'],
    )
    debug_copy_dh_init = f"{WDIR}/outputs/dh/xc_xrayST_soln.p"
    
    shutil.copy(debug_copy_dh_init, f"{p21c.config['direc']}/dh_init_soln.p")
    logging.info(f'Copied dh_init_soln.p from {debug_copy_dh_init}')

    # We have to synchronize at the second step because 21cmFAST acts weird in the first step:
    # - global_params.TK_at_Z_HEAT_MAX is not set correctly (it is probably set and evolved for a step)
    # - global_params.XION_at_Z_HEAT_MAX is not set correctly (it is probably set and evolved for a step)
    # - first step ignores any values added to spin_temp.Tk_box and spin_temp.x_e_box
    z_match = z_edges[1]
    dh_wrapper.evolve(end_rs=(1+z_match)*0.9, rerun=False)
    T_k_DH_init, x_e_DH_init, phot_bath_spec = dh_wrapper.get_init_cond(rs=1+z_match)

    perturbed_field = p21c.perturb_field(redshift=z_edges[1], init_boxes=p21c_initial_conditions)
    spin_temp, ionized_box, brightness_temp = p21c_step(perturbed_field=perturbed_field, spin_temp=None, ionized_box=None, astro_params=astro_params)
    spin_temp.Tk_box += T_k_DH_init - np.mean(spin_temp.Tk_box)
    spin_temp.x_e_box += x_e_DH_init - np.mean(spin_temp.x_e_box)
    ionized_box.xH_box = 1 - spin_temp.x_e_box

    records = []

    #===== main loop =====
    #--- trackers ---
    i_xraycheck_shell_start = 0
    i_xraycheck_bath_start = 0

    z_edges = z_edges[1:] # Maybe fix this later
    z_range = range(len(z_edges)-1)
    if use_tqdm:
        from tqdm import tqdm
        z_range = tqdm(z_range)
    print_str = ''
    dep_tracker = DepTracker()

    #--- loop ---
    for i_z in z_range:

        print_str += f'i_z={i_z}/{len(z_edges)-2} z={z_edges[i_z]:.2f}'

        z_current = z_edges[i_z]
        z_next = z_edges[i_z+1]
        dt = phys.dt_step(z_current, np.exp(abscs['dlnz']))
        
        nBavg = phys.n_B * (1+z_current)**3 # [Bavg / (physical cm)^3]
        delta_plus_one_box = 1 + np.asarray(perturbed_field.density)
        rho_DM_box = delta_plus_one_box * phys.rho_DM * (1+z_current)**3 # [eV/(physical cm)^3]
        x_e_box = np.asarray(spin_temp.x_e_box)
        inj_per_Bavg_box = phys.inj_rate(rho_DM_box, dm_params) * dt * dm_params.struct_boost(1+z_current) / nBavg # [inj/Bavg]
        
        x_e_box_tf = x_e_box
        if debug_unif_delta_tf_param:
            delta_plus_one_box_tf = jnp.full_like(delta_plus_one_box, 1.)
        else:
            delta_plus_one_box_tf = delta_plus_one_box
        tf_wrapper.init_step(
            rs = 1 + z_current,
            delta_plus_one_box = delta_plus_one_box_tf,
            x_e_box = x_e_box_tf,
        )
        
        #===== photon injection and energy deposition =====
        xraycheck_bath_toteng_arr = []
        if not use_21cmfast_xray:

            #----- bath -----
            if 'xc-bath' in debug_flags:

                while phys.conformal_dx_between_z(z_edges[i_xraycheck_bath_start], z_current) >= debug_xray_Rmax_bath:
                    i_xraycheck_bath_start += 1

                print('bath:', i_xraycheck_bath_start, i_xraycheck_shell_start, flush=True)
                
                xraycheck_bath_N_arr = []
                xraycheck_bath_toteng_arr = []

                for i_z_bath in range(i_xraycheck_bath_start, i_xraycheck_shell_start): # uniform injection
                    z_bath = z_edges[i_z_bath]
                    shell_N = np.array(delta_cacher.spectrum_cache.get_spectrum(z_bath).N) # [ph / Msun]

                    if 'xc-custom-SFRD' in debug_flags:
                        cond_sfrd = custom_SFRD
                    else:
                        cond_sfrd = Cond_SFRD_Interpolator
                    emissivity_bracket = get_emissivity_bracket(
                        z=z_bath, delta=0., R=phys.conformal_dx_between_z(z_bath, z_current), dt = dt,
                        debug_nodplus1=debug_nodplus1, cond_sfrd=cond_sfrd, st_sfrd=ST_SFRD_Interpolator,
                    ) # [Msun / Bavg]
                    xraycheck_bath_N_arr.append(shell_N * emissivity_bracket) # [ph / Bavg]
                
                if i_xraycheck_shell_start > i_xraycheck_bath_start: # bath exist

                    xraycheck_bath_N_arr = np.array(xraycheck_bath_N_arr)
                    xraycheck_bath_toteng_arr = np.array([np.dot(N, abscs['photE']) for N in xraycheck_bath_N_arr])
                    xraycheck_bath_N = np.sum(xraycheck_bath_N_arr, axis=0) # [ph / Bavg]

                    dep_tracker.reset(tf_wrapper.dep_box)
                    L_X_bath_spec = Spectrum(abscs['photE'], xraycheck_bath_N, spec_type='N', rs=1+z_current) # [counts / (keV Msun)]
                    tf_wrapper.inject_phot(L_X_bath_spec, inject_type='xray', weight_box=jnp.ones_like(delta_plus_one_box)) # inject bath
                    dep_tracker.record(tf_wrapper.dep_box, from_bath=True)


            print('shell:', i_xraycheck_shell_start, i_z, flush=True)

            #----- older shells -----
            for i_z_shell in range(i_xraycheck_shell_start, i_z):

                delta, L_X_spec, xraycheck_is_box_average, z_donor, R2 = delta_cacher.get_annulus_data(
                    z_current, z_edges[i_z_shell], z_edges[i_z_shell+1]
                )
                
                if 'xc-custom-SFRD' in debug_flags:
                    delta = jnp.array(delta)
                    cond_sfrd = custom_SFRD
                else:
                    delta = np.clip(delta, -1.0+EPSILON, np.max(delta_range)-EPSILON)
                    delta = jnp.array(delta)
                    cond_sfrd = Cond_SFRD_Interpolator
                emissivity_bracket = get_emissivity_bracket(
                    z=z_donor, delta=delta, R=R2, dt = dt,
                    debug_nodplus1=debug_nodplus1, cond_sfrd=cond_sfrd, st_sfrd=ST_SFRD_Interpolator,
                ) # [Msun / Bavg]

                if 'xc-01attenuation' in debug_flags:
                    L_X_spec_inj = L_X_spec.approx_attenuated_spectrum
                    print_str += f'\n    approx attenuation: {L_X_spec.approx_attentuation_arr_repr[xray_i_lo:xray_i_hi]}'
                else:
                    L_X_spec_inj = L_X_spec

                dep_tracker.reset(tf_wrapper.dep_box)
                if np.mean(emissivity_bracket) != 0.:
                    tf_wrapper.inject_phot(L_X_spec_inj, inject_type='xray', weight_box=jnp.asarray(emissivity_bracket))
                dep_tracker.record(tf_wrapper.dep_box, R=phys.conformal_dx_between_z(z_donor, z_current), from_bath=False)

                if xraycheck_is_box_average:
                    i_xraycheck_shell_start = max(i_z_shell+1, i_xraycheck_shell_start)

            #----- new shell can deposits as well! -----
            x_e_for_attenuation = 1 - np.mean(ionized_box.xH_box)
            attenuation_arr = np.array(tf_wrapper.attenuation_arr(rs=1+z_current, x=x_e_for_attenuation)) # convert from jax array
            if 'xc-noatten' in debug_flags: # TMP: turn off attenuation
                attenuation_arr = np.ones_like(attenuation_arr)
            delta_cacher.advance_spectrum(attenuation_arr, z_next) # can handle AttenuatedSpectrum

            L_X_spec_prefac = 1e40 / np.log(4) * u.erg * u.s**-1 * u.M_sun**-1 * u.yr * u.keV**-1 # value in [erg yr / s Msun keV]
            L_X_spec_prefac /= L_X_numerical_factor
            # L_X (E * dN/dE) \propto E^-1
            L_X_dNdE = L_X_spec_prefac.to('1/Msun').value * (abscs['photE']/1000.)**-1 / abscs['photE'] # [1/Msun] * [1/eV] = [1/Msun eV]
            L_X_dNdE[:xray_i_lo] *= 0.
            L_X_dNdE[xray_i_hi:] *= 0.
            L_X_spec = Spectrum(abscs['photE'], L_X_dNdE, spec_type='dNdE', rs=1+z_current) # [1 / Msun eV]
            L_X_spec.switch_spec_type('N') # [1 / Msun]

            L_X_spec.redshift(1+z_next)
            if 'xc-01attenuation' in debug_flags:
                L_X_spec = AttenuatedSpectrum(L_X_spec)
            
            if 'xc-ots' in debug_flags: # before saving, first ots injection
                
                z_donor = z_current
                delta = jnp.array(perturbed_field.density)
                R2 = 0.

                if 'xc-custom-SFRD' in debug_flags:
                    delta = jnp.array(delta)
                    cond_sfrd = custom_SFRD
                else:
                    delta = np.clip(delta, -1.0+EPSILON, np.max(delta_range)-EPSILON)
                    delta = jnp.array(delta)
                    cond_sfrd = Cond_SFRD_Interpolator
                emissivity_bracket = get_emissivity_bracket(
                    z=z_donor, delta=delta, R=R2, dt = dt,
                    debug_nodplus1=debug_nodplus1, cond_sfrd=cond_sfrd, st_sfrd=ST_SFRD_Interpolator,
                ) # [Msun / Bavg]
                
                dep_tracker.reset(tf_wrapper.dep_box)
                if np.mean(emissivity_bracket) != 0.:
                    tf_wrapper.inject_phot(L_X_spec, inject_type='xray', weight_box=jnp.asarray(emissivity_bracket))
                dep_tracker.record(tf_wrapper.dep_box, R=R2, from_bath=False)

                L_X_spec.N *= attenuation_arr

            #----- after possible ots deposition, advance and save -----
            delta_cacher.cache(z_current, jnp.array(perturbed_field.density), L_X_spec)
        
        #===== 21cmFAST step =====
        if i_z > 0: # TEMPORARY: catch NaNs before they go into 21cmFAST
            assert not np.any(np.isnan(input_heating.input_heating)), 'input_heating has NaNs'
            assert not np.any(np.isnan(input_ionization.input_ionization)), 'input_ionization has NaNs'
            assert not np.any(np.isnan(input_jalpha.input_jalpha)), 'input_jalpha has NaNs'
        perturbed_field = p21c.perturb_field(redshift=z_next, init_boxes=p21c_initial_conditions)
        input_heating, input_ionization, input_jalpha = gen_injection_boxes(z_next, p21c_initial_conditions)
        tf_wrapper.populate_injection_boxes(
            input_heating, input_ionization, input_jalpha, dt,
            debug_even_split_f=False,
            ref_depE_per_B=None,
            debug_z = z_current,
            debug_unif_delta_dep = debug_unif_delta_dep,
        )
        spin_temp, ionized_box, brightness_temp = p21c_step(
            perturbed_field, spin_temp, ionized_box,
            input_heating = input_heating,
            input_ionization = input_ionization,
            input_jalpha = input_jalpha,
            astro_params=astro_params
        )
        
        #===== prepare spectra for next step =====
        #--- bath (separating out xray) ---
        prop_phot_N, emit_phot_N = tf_wrapper.prop_phot_N, tf_wrapper.emit_phot_N # propagating and emitted photons have been stored in tf_wrapper up to this point, time to get them out
        # tmp fix 10.2 eV double counting
        prop_phot_N = np.array(prop_phot_N)
        emit_phot_N = np.array(emit_phot_N)
        prop_phot_N[149] = 0.
        emit_bath_N, emit_xray_N = split_xray(emit_phot_N, abscs['photE'])
        phot_bath_spec = Spectrum(abscs['photE'], prop_phot_N + emit_bath_N, rs=1+z_current, spec_type='N') # photons not emitted to the xray band are added to the bath (treated as uniform)
        phot_bath_spec.redshift(1+z_next)

        #--- xray ---
        pass
        
        #===== calculate and save some global quantities =====
        dE_inj_per_Bavg = dm_params.eng_per_inj * np.mean(inj_per_Bavg_box) # [eV per Bavg]
        dE_inj_per_Bavg_unclustered = dE_inj_per_Bavg / dm_params.struct_boost(1+z_current)
        
        record = {
            'z'   : z_next,
            'T_s' : np.mean(spin_temp.Ts_box), # [mK]
            'T_b' : np.mean(brightness_temp.brightness_temp), # [K]
            'T_k' : np.mean(spin_temp.Tk_box), # [K]
            'x_e' : np.mean(spin_temp.x_e_box), # [1]
            '1-x_H' : np.mean(1 - ionized_box.xH_box), # [1]
            'E_phot' : phot_bath_spec.toteng(), # [eV/Bavg]
            'phot_N' : phot_bath_spec.N, # [ph/Bavg]
            'dE_inj_per_B' : dE_inj_per_Bavg,
            'dE_inj_per_Bavg_unclustered' : dE_inj_per_Bavg_unclustered,
            'dep_ion'  : np.mean(tf_wrapper.dep_box[...,0] + tf_wrapper.dep_box[...,1]),
            'dep_exc'  : np.mean(tf_wrapper.dep_box[...,2]),
            'dep_heat' : np.mean(tf_wrapper.dep_box[...,3]),
            'delta_slice' : np.array(perturbed_field.density[0]),
            'x_e_slice' : np.array(spin_temp.x_e_box[0]),
            'x_H_slice' : np.array(ionized_box.xH_box[0]),
            'T_k_slice' : np.array(spin_temp.Tk_box[0]),
            'T_b_slice' : np.array(brightness_temp.brightness_temp[0]),
            'dep_tracker' : {
                'dep_ion_bath' : dep_tracker.dep_ion_bath,
                'dep_heat_bath' : dep_tracker.dep_heat_bath,
                'dep_ion_shells' : np.array(dep_tracker.dep_ion_shells),
                'dep_heat_shells' : np.array(dep_tracker.dep_heat_shells),
                'R_shells' : np.array(dep_tracker.R_shells),
                'bath_toteng_arr' : xraycheck_bath_toteng_arr
            },
            'pc_shell_dep_info': np.mean(spin_temp.SmoothedDelta, axis=(1, 2, 3)),
        }
        records.append(record)
        dep_tracker.clear()

        #===== compare f =====
        f_point = tf_wrapper.phot_dep_tf.point_interp(rs=1+z_current, nBs=1., x=np.mean(spin_temp.x_e_box))
        inj_N = dm_params.inj_phot_spec.N / dm_params.inj_phot_spec.toteng()

        if not use_tqdm:
            # print(print_str, flush=True)
            pass
        print_str = ''
        
    #===== end of loop, save results =====
    arr_records = {k: np.array([r[k] for r in records]) for k in records[0].keys()}
    if save_dir is None:
        save_dir = os.environ['DM21CM_DIR'] + '/outputs/dm21cm'
    np.save(f"{save_dir}/{run_name}_records", arr_records)


#===== utilities for evolve =====

def get_z_edges(z_max, z_min, zplusone_step_factor):
    z_s = [z_min]
    while z_s[-1] < z_max:
        z_s.append((z_s[-1] + 1.0) * zplusone_step_factor - 1.0)
    
    return np.clip(z_s[::-1], None, z_max)


def split_xray(phot_N, phot_eng):
    """Split a photon spectrum (N in bin) into bath and xray band."""

    ex_lo, ex_hi = 1e2, 1e4 # [eV]
    ix_lo = np.searchsorted(phot_eng, ex_lo) # i of first bin greater than ex_lo, excluded
    ix_hi = np.searchsorted(phot_eng, ex_hi) # i of first bin greater than ex_hi, included

    bath_N = np.array(phot_N).copy()
    xray_N = np.array(phot_N).copy()
    bath_N[ix_lo:ix_hi] *= 0
    xray_N[:ix_lo] *= 0
    xray_N[ix_hi:] *= 0
    
    return bath_N, xray_N


def gen_injection_boxes(z_next, p21c_initial_conditions):
    
    input_heating = p21c.input_heating(redshift=z_next, init_boxes=p21c_initial_conditions, write=False)
    input_ionization = p21c.input_ionization(redshift=z_next, init_boxes=p21c_initial_conditions, write=False)
    input_jalpha = p21c.input_jalpha(redshift=z_next, init_boxes=p21c_initial_conditions, write=False)
    
    return input_heating, input_ionization, input_jalpha


def p21c_step(perturbed_field, spin_temp, ionized_box,
             input_heating=None, input_ionization=None, input_jalpha=None, astro_params=None):
    
    spin_temp = p21c.spin_temperature(
        perturbed_field = perturbed_field,
        previous_spin_temp = spin_temp,
        input_heating_box = input_heating,
        input_ionization_box = input_ionization,
        input_jalpha_box = input_jalpha,
        astro_params = astro_params,
    )
    
    ionized_box = p21c.ionize_box(
        perturbed_field = perturbed_field,
        previous_ionize_box = ionized_box,
        spin_temp = spin_temp,
        astro_params = astro_params,
    )
    
    brightness_temp = p21c.brightness_temperature(
        ionized_box = ionized_box,
        perturbed_field = perturbed_field,
        spin_temp = spin_temp,
    )
    
    return spin_temp, ionized_box, brightness_temp


def get_emissivity_bracket(z, delta, R, dt, debug_nodplus1=False, cond_sfrd=None, st_sfrd=None):
    if cond_sfrd is None:
        emissivity_bracket = Cond_SFRD_Interpolator(z, delta, R) # [Msun / Mpc^3 s]
    else:
        emissivity_bracket = cond_sfrd(z, delta, R)
    if np.mean(emissivity_bracket) != 0:
        emissivity_bracket *= (st_sfrd(z) / jnp.mean(emissivity_bracket))
    if not debug_nodplus1:
        emissivity_bracket *= (1 + delta)
    emissivity_bracket *= 1 / (phys.n_B * u.cm**-3).to('Mpc**-3').value * dt # [Msun / Mpc^3 s] * [Bavg / Mpc^3]^-1 * [s] = [Msun / Bavg]
    emissivity_bracket *= L_X_numerical_factor # [Msun / Bavg]
    return emissivity_bracket


# def custom_SFRD(z, delta, r):
#     return jnp.ones_like(delta)

def custom_SFRD(z, delta, r):
    return 1. + delta

# def custom_SFRD(z, delta, r):
#     rv = np.log10(r)
#     sfrd_r_term = np.dot(
#         np.array([r**6, r**5, r**4, r**3, r**2, r, 1.]),
#         np.array([-4.90681604e-10, 6.03851957e-09, -1.61836168e-08,
#                   -7.21151847e-09, 4.31479336e-08, 4.26751313e-08,
#                   1.31003189e-08])
#     )
#     sfrd_d_term = 10**(3.58160894*delta-11.75674659)
#     zv = np.log10(z)
#     sfrd_z_term = np.dot(
#         np.array([z**3, z**2, z, 1.]),
#         np.array([-2.67086153, 6.54275943, -5.86166611, -5.06866749])
#     )
#     return sfrd_r_term * sfrd_d_term * sfrd_z_term

class DepTracker:

    def __init__(self):
        self.clear()
    
    def record(self, dep_box, R=None, from_bath=False):
        dep_ion = np.mean(dep_box[...,0]) + np.mean(dep_box[...,1]) - self.dep_ion_before
        dep_heat = np.mean(dep_box[...,3]) - self.dep_heat_before
        if from_bath:
            self.dep_ion_bath = dep_ion
            self.dep_heat_bath = dep_heat
        else:
            self.dep_ion_shells.append(dep_ion)
            self.dep_heat_shells.append(dep_heat)
            self.R_shells.append(R)
    
    def reset(self, dep_box):
        self.dep_ion_before = np.mean(dep_box[...,0]) + np.mean(dep_box[...,1])
        self.dep_heat_before = np.mean(dep_box[...,3])

    def clear(self):
        self.dep_ion_bath = 0.
        self.dep_heat_bath = 0.
        self.dep_ion_shells = []
        self.dep_heat_shells = []
        self.R_shells = []