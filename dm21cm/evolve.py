"""Main evolution function."""

import os
import sys
import logging
import gc

import numpy as np
from scipy import interpolate
from astropy.cosmology import Planck18
import astropy.units as u

from jax import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp

import py21cmfast as p21c
from py21cmfast import cache_tools

sys.path.append(os.environ['DH_DIR']) # use branch test_dm21cm
from darkhistory.spec.spectrum import Spectrum
from darkhistory.history.reionization import alphaA_recomb
from darkhistory.history.tla import compton_cooling_rate

sys.path.append("..")
import dm21cm.physics as phys
from dm21cm.dh_wrappers import DarkHistoryWrapper, TransferFunctionWrapper
from dm21cm.utils import load_h5_dict
from dm21cm.data_cacher import Cacher
from dm21cm.profiler import Profiler
from dm21cm.spectrum import AttenuatedSpectrum

logging.getLogger().setLevel(logging.INFO)
logging.getLogger('21cmFAST').setLevel(logging.CRITICAL+1)
logging.getLogger('py21cmfast._utils').setLevel(logging.CRITICAL+1)
logging.getLogger('py21cmfast.wrapper').setLevel(logging.CRITICAL+1)


def evolve(run_name, z_start=..., z_end=..., zplusone_step_factor=...,
           dm_params=..., enable_elec=False, tf_version=...,
           p21c_initial_conditions=...,
           use_DH_init=True,
           rerun_DH=False, clear_cache=False,
           use_tqdm=True,
           debug_flags=[],
           debug_xray_multiplier=1.,
           debug_astro_params=None,
           save_dir=None,
           debug_dhc_DH_xe_func=None,
           debug_dhc_delta_fixed=False,
           debug_no_bath=False,
           debug_bath_point_injection=False,
           debug_break_after_z=None,
           dh_bath_N_interp_func=None,
           custom_YHe=None,
           coarsen_interp_factor=None,
           debug_turn_off_pop2ion=False,
           debug_even_split_f=False,
           debug_copy_dh_init=None,
           track_Tk_xe=False,
           track_Tk_xe_set_tf_input=False,
           use_21totf=None,
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
        tf_version (str): Version of DarkHistory transfer function to use.
        p21c_initial_conditions (InitialConditions): Initial conditions from py21cmfast.
        use_DH_init (bool): Whether to use DarkHistory initial conditions.
        
        rerun_DH (bool): Whether to rerun DarkHistory to get initial values.
        clear_cache (bool): Whether to clear cache for 21cmFAST.
        force_reload_tf (bool): Whether to force reload transfer functions. Use when changing dhtf_version.
        use_tqdm (bool): Whether to use tqdm progress bars.
        debug_flags (list): List of debug flags. Can contain:
            'uniform-xray' : Force xray to inject uniformly.
            xraychecks: (Use x_e instead of 1-x_H for tf is now the baseline.)
            'xraycheck' : Xray check mode. And only one of the following:
                'xc-bath' : Xray check: include bath (larger box injection) as well.
                'xc-01attenuation' : Xray check: approximate attenuation.
                'xc-noredshift' : Xray check: don't redshift xrays.
                'xc-noatten' : Xray check: no attenuation.
                'xc-halfatten' : Xray check: half attenuation.
                'xc-force-bath : Xray check: force inject into xray bath.
        debug_astro_params (AstroParams): AstroParams in p21c.
        
        DarkHistory checks:
            debug_dhc_DH_xe_func (callable): Interpolating function to get x_e from DarkHistory.
            debug_dhc_delta_one (bool): Whether to use delta = 1.
        
    Returns:
        dict: Dictionary of results.
    """

    logging.info(f'Using 21cmFAST version {p21c.__version__}')

    #===== cache and memory =====
    p21c.config['direc'] = f"{os.environ['P21C_CACHE_DIR']}/{run_name}"
    logging.info(f"Cache dir: {p21c.config['direc']}")
    os.makedirs(p21c.config['direc'], exist_ok=True)
    if clear_cache:
        cache_tools.clear_cache()
    
    gc.collect()

    #===== initialize =====
    #--- physics parameters ---
    p21c.global_params.Z_HEAT_MAX = z_start + 1e-10
    p21c.global_params.ZPRIME_STEP_FACTOR = zplusone_step_factor
    p21c.global_params.CLUMPING_FACTOR = 1.
    EPSILON = 1e-6
    if custom_YHe is not None:
        p21c.global_params.Y_He = custom_YHe
    if debug_turn_off_pop2ion:
        p21c.global_params.Pop2_ion = 0.

    abscs = load_h5_dict(f"{os.environ['DM21CM_DIR']}/data/abscissas/abscs_{tf_version}.h5")
    if not np.isclose(np.log(zplusone_step_factor), abscs['dlnz']):
        raise ValueError('zplusone_step_factor and tf_version mismatch')
    dm_params.set_inj_specs(abscs)

    box_dim = p21c_initial_conditions.user_params.HII_DIM
    box_len = p21c_initial_conditions.user_params.BOX_LEN
    cosmo = Planck18

    #--- DarkHistory and transfer functions ---
    tf_prefix = f"{os.environ['DM21CM_DATA_DIR']}/tf/{tf_version}"
    tf_wrapper = TransferFunctionWrapper(
        box_dim = box_dim,
        abscs = abscs,
        prefix = tf_prefix,
        enable_elec = enable_elec,
    )

    #--- xray ---
    xray_cacher = Cacher(data_path=f"{p21c.config['direc']}/xray_brightness.h5", cosmo=cosmo, N=box_dim, dx=box_len/box_dim)
    xray_cacher.clear_cache()

    #--- xraycheck ---
    if 'xraycheck' in debug_flags:

        delta_cacher = Cacher(data_path=f"{p21c.config['direc']}/xraycheck_brightness.h5", cosmo=cosmo, N=box_dim, dx=box_len/box_dim, xraycheck=True)
        delta_cacher.clear_cache()

        L_X_numerical_factor = 1e60 # make float happy
        xray_eng_lo = 0.5 * 1000 # [eV]
        xray_eng_hi = 10.0 * 1000 # [eV]
        xray_i_lo = np.searchsorted(abscs['photE'], xray_eng_lo)
        xray_i_hi = np.searchsorted(abscs['photE'], xray_eng_hi)

        res_dict = np.load('../data/xraycheck/Interpolators_0926_2.npz', allow_pickle=True)
        z_range, delta_range, r_range = res_dict['SFRD_Params']

        cond_sfrd_table = res_dict['Cond_SFRD_Table']
        st_sfrd_table =  res_dict['ST_SFRD_Table']

        if coarsen_interp_factor is not None:
            c = coarsen_interp_factor
            z_range = np.concatenate([z_range[::c], z_range[-1:]])
            delta_range = np.concatenate([delta_range[::c], delta_range[-1:]])
            r_range = np.concatenate([r_range[::c], r_range[-1:]])
            cond_sfrd_table = np.concatenate([cond_sfrd_table[::c, :, :], cond_sfrd_table[-1:, :, :]], axis=0)
            cond_sfrd_table = np.concatenate([cond_sfrd_table[:, ::c, :], cond_sfrd_table[:, -1:, :]], axis=1)
            cond_sfrd_table = np.concatenate([cond_sfrd_table[:, :, ::c], cond_sfrd_table[:, :, -1:]], axis=2)
            st_sfrd_table = np.concatenate([st_sfrd_table[::c], st_sfrd_table[-1:]])

        # Takes the redshift as `z`
        # The overdensity parameter smoothed on scale `R`
        # The smoothing scale `R` in units of Mpc
        # Returns the conditional PS star formation rate density in [M_Sun / Mpc^3 / s]
        Cond_SFRD_Interpolator = interpolate.RegularGridInterpolator((z_range, delta_range, r_range), cond_sfrd_table)

        # Takes the redshift as `z`
        # Returns the mean ST star formation rate density star formation rate density in [M_Sun / Mpc^3 / s]
        ST_SFRD_Interpolator = interpolate.interp1d(z_range, st_sfrd_table)

    #--- redshift stepping ---
    z_edges = get_z_edges(z_start, z_end, p21c.global_params.ZPRIME_STEP_FACTOR)

    #--- debug ---
    if use_21totf is not None:
        ref_interp = debug_get_21totf_interp(use_21totf)

    #===== initial steps =====
    dh_wrapper = DarkHistoryWrapper(
        dm_params,
        prefix = p21c.config[f'direc'],
    )
    if debug_copy_dh_init is not None:
        import shutil
        if os.path.exists(debug_copy_dh_init):
            shutil.copy(debug_copy_dh_init, f"{p21c.config['direc']}/dh_init_soln.p")
            logging.info(f'Copied dh_init_soln.p from {debug_copy_dh_init}')
        else:
            logging.warning(f'Could not find dh_init_soln.p at {debug_copy_dh_init}')

    # We have to synchronize at the second step because 21cmFAST acts weird in the first step:
    # - global_params.TK_at_Z_HEAT_MAX is not set correctly (it is probably set and evolved for a step)
    # - global_params.XION_at_Z_HEAT_MAX is not set correctly (it is probably set and evolved for a step)
    # - first step ignores any values added to spin_temp.Tk_box and spin_temp.x_e_box
    z_match = z_edges[1]
    if use_DH_init:
        dh_wrapper.evolve(end_rs=(1+z_match)*0.9, rerun=rerun_DH)
        T_k_DH_init, x_e_DH_init = dh_wrapper.get_init_cond(rs=1+z_match)
        phot_bath_spec = dh_wrapper.get_phot_bath(rs=1+z_match)
    else:
        phot_bath_spec = Spectrum(abscs['photE'], np.zeros_like(abscs['photE']), spec_type='N', rs=1+z_match) # [ph / Bavg]
    if debug_no_bath:
        logging.warning('Turning off bath, remember to turn back on')
        phot_bath_spec *= 0.

    perturbed_field = p21c.perturb_field(redshift=z_edges[1], init_boxes=p21c_initial_conditions)
    spin_temp, ionized_box, brightness_temp = p21c_step(perturbed_field=perturbed_field, spin_temp=None, ionized_box=None, astro_params=debug_astro_params)
    if use_DH_init:
        spin_temp.Tk_box += T_k_DH_init - np.mean(spin_temp.Tk_box)
        spin_temp.x_e_box += x_e_DH_init - np.mean(spin_temp.x_e_box)
        ionized_box.xH_box = 1 - spin_temp.x_e_box

    records = []
    record = {
        'z'   : z_edges[1],
        'T_s' : np.mean(spin_temp.Ts_box), # [mK]
        'T_b' : np.mean(brightness_temp.brightness_temp), # [K]
        'T_k' : np.mean(spin_temp.Tk_box), # [K]
        'x_e' : np.mean(spin_temp.x_e_box), # [1]
        '1-x_H' : np.mean(1 - ionized_box.xH_box), # [1]
        'E_phot' : phot_bath_spec.toteng(), # [eV/Bavg]
        'phot_N' : phot_bath_spec.N, # [ph/Bavg]
        #'injected_bath_N' : np.zeros_like(phot_bath_spec.N), # [ph/Bavg]
        'dE_inj_per_B' : 0.,
        'dE_inj_per_Bavg_unclustered' : 0.,
        'dep_ion'  : 0.,
        'dep_exc'  : 0.,
        'dep_heat' : 0.,
        'x_e_slice' : np.array(spin_temp.x_e_box[10]),
        'x_H_slice' : np.array(ionized_box.xH_box[10]),
    }
    if track_Tk_xe:
        T_k_track = np.mean(spin_temp.Tk_box)
        x_e_track = np.mean(spin_temp.x_e_box)
        record.update({
            'T_k_tracker' : T_k_track, # [K]
            'x_e_tracker' : x_e_track, # [1]
        })
    records.append(record)


    #===== main loop =====
    #--- trackers ---
    if 'xraycheck' in debug_flags:
        i_xraycheck_loop_start = 0
    else:
        i_xray_loop_start = 0 # where we start looking for annuli
    
    profiler = Profiler()

    z_edges = z_edges[1:] # Maybe fix this later
    z_iterator = range(len(z_edges)-1)
    if use_tqdm:
        from tqdm import tqdm
        z_iterator = tqdm(z_iterator)
    print_str = ''

    #--- loop ---
    for i_z in z_iterator:
        print(f'i_z={i_z}/{len(z_edges)-2} z={z_edges[i_z]:.2f}', file=sys.stderr, flush=True)
        print_str += f'i_z={i_z}/{len(z_edges)-2} z={z_edges[i_z]:.2f}'

        z_current = z_edges[i_z]
        z_next = z_edges[i_z+1]
        # dt = ( cosmo.age(z_next) - cosmo.age(z_current) ).to('s').value # cosmo
        # dt = np.interp(1+z_current, abscs['rs'], dts[:,1])
        dt = phys.dt_step(z_current, np.exp(abscs['dlnz']))
        
        nBavg = phys.n_B * (1+z_current)**3 # [Bavg / (physical cm)^3]
        delta_plus_one_box = 1 + np.asarray(perturbed_field.density)
        rho_DM_box = delta_plus_one_box * phys.rho_DM * (1+z_current)**3 # [eV/(physical cm)^3]
        if 'xraycheck' in debug_flags:
            x_e_box = np.asarray(spin_temp.x_e_box)
        else:
            # logging.warning('\nusing xe = 1 - mean(ionized_box.xH_box) for deposition\n')
            x_e_box = np.asarray(1 - ionized_box.xH_box)
        inj_per_Bavg_box = phys.inj_rate(rho_DM_box, dm_params) * dt * dm_params.struct_boost(1+z_current) / nBavg # [inj/Bavg]
        
        if debug_dhc_DH_xe_func is not None:
            x_e_box_tf = jnp.full_like(x_e_box, debug_dhc_DH_xe_func(z_current))
        else:
            x_e_box_tf = x_e_box
        if debug_dhc_delta_fixed:
            delta_plus_one_box_tf = jnp.full_like(delta_plus_one_box, 1.)
        else:
            delta_plus_one_box_tf = delta_plus_one_box
        if track_Tk_xe_set_tf_input:
            x_e_box_tf = jnp.full_like(x_e_box, x_e_track)
            logging.warning(f'Setting x_e_box_tf to x_e_track.')
            #delta_plus_one_box_tf = jnp.full_like(delta_plus_one_box, 1.)
        tf_wrapper.init_step(
            rs = 1 + z_current,
            delta_plus_one_box = delta_plus_one_box_tf,
            x_e_box = x_e_box_tf,
        )
        
        #===== photon injection and energy deposition =====
        #--- xray ---
        profiler.start()
        if 'xraycheck' in debug_flags:
            if 'xc-bath' in debug_flags:
                xraycheck_bath_N = np.zeros((500,)) # [ph / Bavg]
                emissivity_bracket_unif = 0.
                if 'xc-force-bath' in debug_flags:
                    i_xraycheck_loop_start = i_z # all goes into uniform injection
                for i_z_shell in range(i_xraycheck_loop_start): # uniform injection
                    z_shell = z_edges[i_z_shell]
                    shell_N = np.array(delta_cacher.spectrum_cache.get_spectrum(z_shell).N) # [ph / Msun]
                    print(f'XCBATH-DEBUG: BATH i_z={i_z} i_shell={i_z_shell}, raw {np.dot(shell_N, abscs["photE"]):.3e} eV/Msun')

                    delta_unif = 0. # just a number
                    emissivity_bracket_unif = Cond_SFRD_Interpolator((z_shell, delta_unif, 512.-EPSILON)) # [Msun / Mpc^3 s]
                    if np.mean(emissivity_bracket_unif) > 0:
                        emissivity_bracket_unif *= (ST_SFRD_Interpolator(z_shell) / np.mean(emissivity_bracket_unif)) # [Msun / Mpc^3 s]
                    emissivity_bracket_unif *= (1 + delta_unif) / (phys.n_B * u.cm**-3).to('Mpc**-3').value * dt # [Msun / Mpc^3 s] * [Bavg / Mpc^3]^-1 * [s] = [Msun / Bavg]
                    emissivity_bracket_unif *= L_X_numerical_factor * debug_xray_multiplier # [Msun / Bavg]
                    shell_N *= emissivity_bracket_unif # [ph / Bavg]
                    xraycheck_bath_N += shell_N # put in bath

                    print(f'XCBATH-DEBUG: BATH i_z={i_z} i_shell={i_z_shell}, inject {np.dot(shell_N, abscs["photE"]):.3e} eV/Bavg')

                L_X_bath_spec = Spectrum(abscs['photE'], xraycheck_bath_N, spec_type='N', rs=1+z_current) # [counts / (keV Msun)]
                weight = jnp.ones_like(delta_plus_one_box)
                tf_wrapper.inject_phot(L_X_bath_spec, inject_type='xray', weight_box=weight) # inject bath

                print_str += f' i_xray_bath=0-{i_xraycheck_loop_start}'
                print_str += f' L_X_bath eng={L_X_bath_spec.toteng():.3e} eV/Bavg'
                profiler.record('xraycheck bath')
            
            emissivity_bracket = 0.

            print_str += f' delta-mean='

            for i_z_shell in range(i_xraycheck_loop_start, i_z):

                delta, L_X_spec, xraycheck_is_box_average, z_donor, R2 = delta_cacher.get_annulus_data(
                    z_current, z_edges[i_z_shell], z_edges[i_z_shell+1]
                )
                print(f'XCBATH-DEBUG: SHELL i_z={i_z} i_shell={i_z_shell}, raw {np.dot(L_X_spec.N, abscs["photE"]):.3e} eV/Msun')
                delta = np.clip(delta, -1.0+EPSILON, 1.5-EPSILON)
                delta = np.array(delta)
                print_str += f' {np.mean(delta):.3f}'
                emissivity_bracket = Cond_SFRD_Interpolator((z_donor, delta, R2))
                if np.mean(emissivity_bracket) > 0:
                    emissivity_bracket *= (ST_SFRD_Interpolator(z_donor) / np.mean(emissivity_bracket))
                z_shell = z_edges[i_z_shell]
                emissivity_bracket *= (1 + delta) / (phys.n_B * u.cm**-3).to('Mpc**-3').value * dt
                emissivity_bracket *= L_X_numerical_factor * debug_xray_multiplier
                if xraycheck_is_box_average:
                    i_xraycheck_loop_start = max(i_z_shell+1, i_xraycheck_loop_start)

                if 'xc-01attenuation' in debug_flags:
                    L_X_spec_inj = L_X_spec.approx_attenuated_spectrum
                    print_str += f'\n    approx attenuation: {L_X_spec.approx_attentuation_arr_repr[xray_i_lo:xray_i_hi]}'
                else:
                    L_X_spec_inj = L_X_spec
                
                print(f'XCBATH-DEBUG: SHELL i_z={i_z} i_shell={i_z_shell}, inject {np.mean(emissivity_bracket)*np.dot(L_X_spec_inj.N, abscs["photE"]):.3e} eV/Bavg')

                if ST_SFRD_Interpolator(z_donor) > 0.:
                    tf_wrapper.inject_phot(L_X_spec_inj, inject_type='xray', weight_box=jnp.asarray(emissivity_bracket))
            
            print_str += f' shells:{i_xraycheck_loop_start}-{i_z}'
            if i_z > i_xraycheck_loop_start: # if shells were injected at all
                avg_eng = np.mean(emissivity_bracket)*L_X_spec_inj.toteng()
                print_str += f' shell xray:{avg_eng:.3e} eV/Bavg'
            profiler.record('xraycheck')

        else: # regular routine
            for i_z_shell in range(i_xray_loop_start, i_z):

                xray_brightness_box, xray_spec, is_box_average = xray_cacher.get_annulus_data(
                    z_current, z_edges[i_z_shell], z_edges[i_z_shell+1]
                )
                # If we are smoothing on the scale of the box then dump to the global bath spectrum.
                # The deposition will happen with `phot_bath_spec`, and we will not revisit this shell.
                if is_box_average or 'uniform_xray' in debug_flags:
                    #phot_bath_spec.N += xray_brightness_box[0, 0, 0] * xray_spec.N # TMP: fix the [0,0,0]
                    phot_bath_spec.N += xray_spec.N
                    i_xray_loop_start = max(i_z_shell+1, i_xray_loop_start)
                else:
                    print(f'DEBUG uniform-xray: xray_brightness_box = {np.mean(xray_brightness_box):.3e}')
                    tf_wrapper.inject_phot(xray_spec, inject_type='xray', weight_box=xray_brightness_box)

            profiler.record('xray')

            #--- bath and homogeneous portion of xray ---
            if debug_no_bath:
                phot_bath_spec *= 0.
            if debug_bath_point_injection:
                #if np.isclose(z_current, 37.713184, rtol=1e-3): # 38.713184 test
                if np.isclose(z_current, 4.530668e+01, rtol=1e-3):
                    logging.warning(f'Point injecting bath at z={z_current} ---------------------')
                    phot_bath_spec.N *= 0.
                    phot_bath_spec.N[407] = 1e-5
                    print(f'bath energy', phot_bath_spec.toteng())
                    print(f'eng per inj', dm_params.inj_phot_spec.toteng())
                    print(f'inj_per_Bavg', np.mean(inj_per_Bavg_box))
                    print(f'inj eng', dm_params.inj_phot_spec.toteng() * np.mean(inj_per_Bavg_box))
                    print(np.where(phot_bath_spec.N > 0))
                    print(np.where(dm_params.inj_phot_spec.N > 0))
            if dh_bath_N_interp_func is not None:
                phot_bath_spec.N = dh_bath_N_interp_func(z_current)
            print_str += f' bath.toteng={phot_bath_spec.toteng():.3e} eV/Bavg'
            injected_bath_N = np.array(phot_bath_spec.N)
            tf_wrapper.inject_phot(phot_bath_spec, inject_type='bath')
            
            #--- dark matter (on-the-spot) ---
            tf_wrapper.inject_from_dm(dm_params, inj_per_Bavg_box)

            profiler.record('bath+dm')
        
        #===== 21cmFAST step =====
        if i_z > 0: # TEMPORARY: catch NaNs before they go into 21cmFAST
            if np.any(np.isnan(input_heating.input_heating)):
                raise ValueError('input_heating.input_heating has NaNs')
            if np.any(np.isnan(input_ionization.input_ionization)):
                raise ValueError('input_ionization.input_ionization has NaNs')
            if np.any(np.isnan(input_jalpha.input_jalpha)):
                raise ValueError('input_jalpha.input_jalpha has NaNs')
        perturbed_field = p21c.perturb_field(redshift=z_next, init_boxes=p21c_initial_conditions)
        input_heating, input_ionization, input_jalpha = gen_injection_boxes(z_next, p21c_initial_conditions)
        if use_21totf:
            ref_depE_per_B = ref_interp(z_current) * phys.A_per_B
        else:
            ref_depE_per_B = None
        tf_wrapper.populate_injection_boxes(
            input_heating, input_ionization, input_jalpha, dt,
            debug_even_split_f=debug_even_split_f,
            ref_depE_per_B=ref_depE_per_B,
        )
        # print('before', np.mean(spin_temp.Tk_box), np.mean(spin_temp.x_e_box), flush=True)
        # print('input_heating', np.mean(input_heating.input_heating), flush=True)
        spin_temp, ionized_box, brightness_temp = p21c_step(
            perturbed_field, spin_temp, ionized_box,
            input_heating = input_heating,
            input_ionization = input_ionization,
            input_jalpha = input_jalpha,
            astro_params=debug_astro_params
        )
        # print('after', np.mean(spin_temp.Tk_box), np.mean(spin_temp.x_e_box), flush=True)

        if track_Tk_xe:
            T_k_track += np.mean(input_heating.input_heating)
            x_e_track += np.mean(input_ionization.input_ionization)

            T = T_k_track
            x = x_e_track
            z = z_current
            rs = 1 + z

            dxion_sink_dt = - alphaA_recomb('HII', phys.kB*T) * x**2 * phys.n_H * (1+z)**3
            x_e_track += dxion_sink_dt * dt

            dTdz_adia = 2 * T / (1 + z)
            #dcomp_dzp = dcomp_dzp_prefactor * (x_e_track/(1 + x_e_track)) * ( Trad_fast - T ); # 21cmFAST
            dTdz_comp_DH = compton_cooling_rate(x, phys.chi*x, 0, phys.kB*T, rs) * phys.dtdz(rs) / phys.kB / (3/2 * phys.n_H * (rs**3) * (1 + phys.chi + x))
            T_k_track += (dTdz_adia + dTdz_comp_DH) * (z_next - z)



        profiler.record('21cmFAST')
        
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
        if debug_no_bath:
            phot_bath_spec *= 0.

        #--- xray ---
        if 'xraycheck' in debug_flags:
            x_e_for_attenuation = 1 - np.mean(ionized_box.xH_box)
            attenuation_arr = np.array(tf_wrapper.attenuation_arr(rs=1+z_current, x=x_e_for_attenuation)) # convert from jax array
            if 'xc-halfatten' in debug_flags: # TMP: half attenuation
                attenuation_arr = 1 - (1 - attenuation_arr) / 2
            if 'xc-noatten' in debug_flags: # TMP: turn off attenuation
                attenuation_arr = np.ones_like(attenuation_arr)
            delta_cacher.advance_spectrum(attenuation_arr, z_next, noredshift=('xc-noredshift' in debug_flags)) # can handle AttenuatedSpectrum

            print_str += f" atten. mean={np.mean(attenuation_arr):.4f}"

            L_X_spec_prefac = 1e40 / np.log(4) * u.erg * u.s**-1 * u.M_sun**-1 * u.yr * u.keV**-1 # value in [erg yr / s Msun keV]
            L_X_spec_prefac /= L_X_numerical_factor
            # L_X (E * dN/dE) \propto E^-1
            L_X_dNdE = L_X_spec_prefac.to('1/Msun').value * (abscs['photE']/1000.)**-1 / abscs['photE'] # [1/Msun] * [1/eV] = [1/Msun eV]
            L_X_dNdE[:xray_i_lo] *= 0.
            L_X_dNdE[xray_i_hi:] *= 0.
            L_X_spec = Spectrum(abscs['photE'], L_X_dNdE, spec_type='dNdE', rs=1+z_current) # [1 / Msun eV]
            L_X_spec.switch_spec_type('N') # [1 / Msun]

            if 'xc-noredshift' in debug_flags:
                L_X_spec.rs = 1+z_next
            else:
                L_X_spec.redshift(1+z_next)

            if 'xc-01attenuation' in debug_flags:
                L_X_spec = AttenuatedSpectrum(L_X_spec)
            delta_cacher.cache(z_current, perturbed_field.density, L_X_spec)
        
        else:
            x_e_for_attenuation = 1 - np.mean(ionized_box.xH_box)
            attenuation_arr = np.array(tf_wrapper.attenuation_arr(rs=1+z_current, x=np.mean(x_e_for_attenuation))) # convert from jax array
            xray_cacher.advance_spectrum(attenuation_arr, z_next)

            xray_spec = Spectrum(abscs['photE'], emit_xray_N, rs=1+z_current, spec_type='N') # [ph / Bavg]
            xray_spec.redshift(1+z_next)
            xray_tot_eng = np.dot(abscs['photE'], emit_xray_N)
            # print(f'DM21CM DEBUG: xray_tot_eng={xray_tot_eng:.3e} eV/Bavg')
            if xray_tot_eng == 0.:
                xray_rel_eng_box = np.zeros_like(tf_wrapper.xray_eng_box)
            else:
                xray_rel_eng_box = tf_wrapper.xray_eng_box / xray_tot_eng # [1 (relative energy) / Bavg]
            # print(f'DM21CM DEBUG: xray_rel_eng_box={np.mean(xray_rel_eng_box):.3e} 1/Bavg')
            xray_cacher.cache(z_current, xray_rel_eng_box, xray_spec)
        
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
            #'injected_bath_N' : injected_bath_N, # [ph/Bavg]
            'dE_inj_per_B' : dE_inj_per_Bavg,
            'dE_inj_per_Bavg_unclustered' : dE_inj_per_Bavg_unclustered,
            'dep_ion'  : np.mean(tf_wrapper.dep_box[...,0] + tf_wrapper.dep_box[...,1]),
            'dep_exc'  : np.mean(tf_wrapper.dep_box[...,2]),
            'dep_heat' : np.mean(tf_wrapper.dep_box[...,3]),
            'x_e_slice' : np.array(spin_temp.x_e_box[10]),
            'x_H_slice' : np.array(ionized_box.xH_box[10]),
        }
        if track_Tk_xe:
            record.update({
                'T_k_tracker' : T_k_track, # [K]
                'x_e_tracker' : x_e_track, # [1]
            })
        records.append(record)

        profiler.record('prep_next')

        #===== compare f =====
        f_point = tf_wrapper.phot_dep_tf.point_interp(rs=1+z_current, nBs=1., x=np.mean(spin_temp.x_e_box))
        inj_N = dm_params.inj_phot_spec.N / dm_params.inj_phot_spec.toteng()
        # print('----- DM21CM -----')
        # print('z', z_current)
        # print(np.dot(inj_N, f_point))
        # print(np.mean(tf_wrapper.dep_box[...,0]) / phys.A_per_B, 'eV/A')
        # print((np.mean(tf_wrapper.dep_box[...,1]) + np.mean(tf_wrapper.dep_box[...,2])) / phys.A_per_B, 'eV/A')
        # print(np.mean(tf_wrapper.dep_box[...,3]) / phys.A_per_B, 'eV/A')
        # print('-----------------', flush=True)

        if not use_tqdm:
            # print(print_str, flush=True)
            pass
        print_str = ''
        if debug_break_after_z is not None and z_current < debug_break_after_z:
            break
        
    #===== end of loop, save results =====
    arr_records = {k: np.array([r[k] for r in records]) for k in records[0].keys()}
    if save_dir is None:
        save_dir = os.environ['DM21CM_DIR'] + '/outputs/dm21cm'
    np.save(f"{save_dir}/{run_name}_records", arr_records)

    profiler.print_summary()

    return {
        'profiler' : profiler,
    }


#===== utilities for evolve =====

def get_z_edges(z_max, z_min, zplusone_step_factor):
    z_s = [z_min]
    while z_s[-1] < z_max:
        z_s.append((z_s[-1] + 1.0) * zplusone_step_factor - 1.0)
    
    return np.clip(z_s[::-1], None, z_max)


def split_xray(phot_N, phot_eng):
    """Split a photon spectrum (N in bin) into bath and xray band."""

    ex_lo, ex_hi = 5e2, 1e4 # [eV]
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


def debug_get_21totf_interp(fn):
    line_arr = []
    z_arr = []
    e_arr = []
    with open(fn, 'r') as f:
        for line in f:
            if "E_(tot=heat+ion+lya) =" in line:
                line_arr.append(line)
                tokens = line.split()
                e_arr.append(float(tokens[-2]))
            if "TsBox.c DEBUG:" in line:
                line_arr.append(line)
                tokens = line.split()
                z_arr.append(float(tokens[-1]))
    assert len(z_arr) == len(e_arr)
    return interpolate.interp1d(z_arr, e_arr, kind='linear', bounds_error=False, fill_value='extrapolate')