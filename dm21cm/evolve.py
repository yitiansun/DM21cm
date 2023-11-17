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

sys.path.append(os.environ['DH_DIR'])
from darkhistory.spec.spectrum import Spectrum

sys.path.append(os.environ['DM21CM_DIR'])
import dm21cm.physics as phys
from dm21cm.dh_wrappers import DarkHistoryWrapper, TransferFunctionWrapper
from dm21cm.utils import load_h5_dict
from dm21cm.data_cacher import Cacher
from dm21cm.profiler import Profiler

logging.getLogger().setLevel(logging.INFO)
logging.getLogger('21cmFAST').setLevel(logging.CRITICAL+1)
logging.getLogger('py21cmfast._utils').setLevel(logging.CRITICAL+1)
logging.getLogger('py21cmfast.wrapper').setLevel(logging.CRITICAL+1)


def evolve(run_name,
           z_start=..., z_end=...,
           dm_params=..., enable_elec=False,
           p21c_initial_conditions=...,
           p21c_astro_params=None,
           use_DH_init=True, rerun_DH=False,
           clear_cache=False,
           use_tqdm=True,
           tf_on_device=True,
           no_injection=False,
           subcycle_factor=1,
           max_n_shell=None,
           ):
    """
    Main evolution function.

    Args:
        run_name (str):             Name of run. Used for cache directory.
        z_start (float):            Starting redshift.
        z_end (float):              Ending redshift.
        subcycle_factor (int):      Number of subcycles per 21cmFAST step.
        max_n_shell (int or None):  Number total shells used in xray injection. If None, use all shells smaller than the box size.
        dm_params (dm21cm.dm_params.DMParams):             Dark matter (DM) parameters.
        enable_elec (bool):                                Whether to enable electron injection.
        p21c_initial_conditions (p21c.InitialConditions):  Initial conditions for 21cmFAST.
        p21c_astro_params (p21c.AstroParams):              AstroParams for 21cmFAST.
        use_DH_init (bool):         Whether to use DarkHistory initial conditions.
        rerun_DH (bool):            Whether to rerun DarkHistory to get initial values.
        clear_cache (bool):         Whether to clear cache for 21cmFAST.
        use_tqdm (bool):            Whether to use tqdm progress bars.
        tf_on_device (bool):        Whether to put transfer functions on device (GPU).
        no_injection (bool):        Whether to skip injection and energy deposition.

    Returns:
        dict: Dictionary of results.
    """

    logging.info(f'Using 21cmFAST version {p21c.__version__}')

    #===== data and cache =====
    data_dir = os.environ['DM21CM_DATA_DIR']
    cache_dir = os.environ['P21C_CACHE_DIR'] + '/' + run_name
    p21c.config['direc'] = cache_dir
    logging.info(f"Cache dir: {cache_dir}")
    os.makedirs(cache_dir, exist_ok=True)
    if clear_cache:
        cache_tools.clear_cache()
    gc.collect()

    #===== initialize =====
    #--- physics parameters ---
    abscs = load_h5_dict(f"{data_dir}/abscissas.h5")
    dm_params.set_inj_specs(abscs)

    EPSILON = 1e-6
    p21c.global_params.Z_HEAT_MAX = z_start + EPSILON
    p21c.global_params.ZPRIME_STEP_FACTOR = abscs['zplusone_step_factor'] ** subcycle_factor

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

    #--- xray ---
    xray_cacher = Cacher(data_path=f"{cache_dir}/xray_brightness.h5", cosmo=cosmo, N=box_dim, dx=box_len/box_dim)
    xray_cacher.clear_cache()

    #--- redshift stepping ---
    z_edges = get_z_edges(z_start, z_end, abscs['zplusone_step_factor'])
    z_edges_coarse = get_z_edges(z_start, z_end, p21c.global_params.ZPRIME_STEP_FACTOR)

    #===== initial steps =====
    dh_wrapper = DarkHistoryWrapper(dm_params, prefix=p21c.config[f'direc'])

    # We have to synchronize at the second step because 21cmFAST acts weird in the first step:
    # - global_params.TK_at_Z_HEAT_MAX is not set correctly (it is probably set and evolved for a step)
    # - global_params.XION_at_Z_HEAT_MAX is not set correctly (it is probably set and evolved for a step)
    # - first step ignores any values added to spin_temp.Tk_box and spin_temp.x_e_box
    z_match = z_edges_coarse[1]
    if use_DH_init:
        dh_wrapper.evolve(end_rs=(1+z_match)*0.9, rerun=rerun_DH)
        T_k_DH_init, x_e_DH_init, phot_bath_spec = dh_wrapper.get_init_cond(rs=1+z_match)
    else:
        phot_bath_spec = Spectrum(abscs['photE'], np.zeros_like(abscs['photE']), spec_type='N', rs=1+z_match) # [ph / Bavg]

    perturbed_field = p21c.perturb_field(redshift=z_edges[1], init_boxes=p21c_initial_conditions, write = True)
    spin_temp, ionized_box, brightness_temp = p21c_step(perturbed_field=perturbed_field, spin_temp=None, ionized_box=None, astro_params=p21c_astro_params)
    if use_DH_init:
        spin_temp.Tk_box += T_k_DH_init - np.mean(spin_temp.Tk_box)
        spin_temp.x_e_box += x_e_DH_init - np.mean(spin_temp.x_e_box)
        ionized_box.xH_box = 1 - spin_temp.x_e_box

    #===== main loop =====
    # advance z_edges to start with z_match
    while not np.isclose(z_edges[0], z_match):
        z_edges = z_edges[1:]
    z_edges_coarse = z_edges_coarse[1:]
    z_range = range(len(z_edges)-1)
    records = []
    if use_tqdm:
        from tqdm import tqdm
        z_range = tqdm(z_range)
    profiler = Profiler()

    #--- trackers ---
    i_xray_loop_start = 0 # where we start looking for annuli

    #--- loop ---
    for i_z in z_range:

        profiler.start()
        print_str = f'i_z={i_z}/{len(z_edges)-2} z={z_edges[i_z]:.2f}'
        i_z_coarse = i_z // subcycle_factor

        #===== physical quantities =====
        z_current = z_edges[i_z]
        z_next = z_edges[i_z+1]
        dt = phys.dt_step(z_current, np.exp(abscs['dlnz']))

        #--- for interpolation ---
        delta_plus_one_box = 1 + np.asarray(perturbed_field.density)
        x_e_box = np.asarray(1 - ionized_box.xH_box)
        T_k_box = np.asarray(spin_temp.Tk_box)
        tf_wrapper.set_params(rs=1+z_current, delta_plus_one_box=delta_plus_one_box, x_e_box=x_e_box, T_k_box=T_k_box)
        tf_wrapper.reset_phot() # reset photon each step, but deposition is reset only after populating boxes

        #--- for dark matter ---
        nBavg = phys.n_B * (1+z_current)**3 # [Bavg / (physical cm)^3]
        rho_DM_box = delta_plus_one_box * phys.rho_DM * (1+z_current)**3 # [eV/(physical cm)^3]
        inj_per_Bavg_box = phys.inj_rate(rho_DM_box, dm_params) * dt * dm_params.struct_boost(1+z_current) / nBavg # [inj/Bavg]

        
        if not no_injection:

            #===== photon injection and energy deposition =====
            #--- xray ---
            # First we dump to bath all spectra whose corresponding shells are larger than the box size.
            while i_xray_loop_start < i_z:
                z_shell_start = z_edges[i_xray_loop_start]
                z_shell_end = z_edges[i_xray_loop_start+1]
                r_start = phys.conformal_dx_between_z(z_current, z_shell_start) # [cMpc]
                r_end = phys.conformal_dx_between_z(z_current, z_shell_end) # [cMpc]
                
                if min(r_start, r_end) > box_len/2: # if the shell is larger than the box size
                    phot_bath_spec.N += xray_spec.N
                    i_xray_loop_start += 1
                else:
                    break

            # Then we select the chosen shell indices for deposition
            if max_n_shell is not None:
                i_max = i_z - i_xray_loop_start
                inds_increasing = geom_inds(i_max=i_max, i_transition=10, n_goal=max_n_shell)
                inds_chosen_shells = i_z - inds_increasing
            else:
                inds_chosen_shells = list(range(i_xray_loop_start, i_z)) # all shells smaller than the box size are chosen

            # Finally, we accumulate spectra from the non-chosen shells and deposit only on the chosen shells
            accumulated_shell_spec = Spectrum(abscs['photE'], np.zeros_like(abscs['photE']), spec_type='N', rs=1+z_current) # [ph/Bavg]

            for i_z_shell in range(i_xray_loop_start, i_z):
                z_shell_start = z_edges[i_z_shell]
                z_shell_end = z_edges[i_z_shell+1]

                if i_z_shell not in inds_chosen_shells:
                    accumulated_shell_spec += xray_cacher.spectrum_cache.get_spectrum(z_shell_start)
                    continue

                xray_brightness_box, xray_spec, _ = xray_cacher.get_annulus_data(z_current, z_shell_start, z_shell_end)
                xray_spec += accumulated_shell_spec
                tf_wrapper.inject_phot(xray_spec, inject_type='xray', weight_box=xray_brightness_box)

                accumulated_shell_spec *= 0.

            profiler.record('xray')

            #--- bath and homogeneous portion of xray ---
            tf_wrapper.inject_phot(phot_bath_spec, inject_type='bath')

            #--- dark matter (on-the-spot) ---
            tf_wrapper.inject_from_dm(dm_params, inj_per_Bavg_box)

            profiler.record('bath+dm')

            #===== prepare spectra for next step =====
            #--- bath (separating out xray) ---
            prop_phot_N = np.array(tf_wrapper.prop_phot_N) # propagating and emitted photons have been stored in tf_wrapper up to this point, time to get them out
            emit_phot_N = np.array(tf_wrapper.emit_phot_N)
            emit_bath_N, emit_xray_N = split_xray(emit_phot_N, abscs['photE'])
            phot_bath_spec = Spectrum(abscs['photE'], prop_phot_N + emit_bath_N, rs=1+z_current, spec_type='N') # photons not emitted to the xray band are added to the bath (treated as uniform)
            phot_bath_spec.redshift(1+z_next)

            #--- xray ---
            x_e_for_attenuation = 1 - np.mean(ionized_box.xH_box)
            attenuation_arr = np.array(tf_wrapper.attenuation_arr(rs=1+z_current, x=np.mean(x_e_for_attenuation))) # convert from jax array
            xray_cacher.advance_spectrum(attenuation_arr, z_next)

            xray_spec = Spectrum(abscs['photE'], emit_xray_N, rs=1+z_current, spec_type='N') # [ph/Bavg]
            xray_spec.redshift(1+z_next)
            if np.mean(tf_wrapper.xray_eng_box) != 0.:
                # dont' normalize w.r.t. to np.dot(abscs['photE'], emit_xray_N) because
                # that contains not only the emission but propagation
                xray_rel_eng_box = tf_wrapper.xray_eng_box / jnp.mean(tf_wrapper.xray_eng_box) # [1 (relative energy)/Bavg]
            else:
                xray_rel_eng_box = np.zeros_like(tf_wrapper.xray_eng_box) # [1 (relative energy)/Bavg]
            
            xray_cacher.cache(z_current, xray_rel_eng_box, xray_spec)

            profiler.record('prep_next')

        #===== 21cmFAST step =====
        # check if z_next matches
        if (i_z_coarse + 1) * subcycle_factor == (i_z + 1):

            #print(f'evolves 21cmFAST at i_z={i_z}, i_z_coarse={i_z_coarse}, z_next={z_next:.2f}={z_edges_coarse[i_z_coarse+1]:.2f}')
            assert np.isclose(z_next, z_edges_coarse[i_z_coarse+1]) # cross check remove later

            perturbed_field = p21c.perturb_field(redshift=z_next, init_boxes=p21c_initial_conditions)
            input_heating, input_ionization, input_jalpha = gen_injection_boxes(z_next, p21c_initial_conditions)
            tf_wrapper.populate_injection_boxes(input_heating, input_ionization, input_jalpha, dt,)
            spin_temp, ionized_box, brightness_temp = p21c_step(
                perturbed_field, spin_temp, ionized_box,
                input_heating = input_heating,
                input_ionization = input_ionization,
                input_jalpha = input_jalpha,
                astro_params = p21c_astro_params
            )

            profiler.record('21cmFAST')

            #===== calculate and save some quantities =====
            dE_inj_per_Bavg = dm_params.eng_per_inj * np.mean(inj_per_Bavg_box) # [eV/Bavg]
            dE_inj_per_Bavg_unclustered = dE_inj_per_Bavg / dm_params.struct_boost(1+z_current) # [eV/Bavg]

            records.append({
                'z'   : z_next,
                'T_s' : np.mean(spin_temp.Ts_box), # [mK]
                'T_b' : np.mean(brightness_temp.brightness_temp), # [K]
                'T_k' : np.mean(spin_temp.Tk_box), # [K]
                'x_e' : np.mean(spin_temp.x_e_box), # [1]
                '1-x_H' : np.mean(1 - ionized_box.xH_box), # [1]
                'E_phot' : phot_bath_spec.toteng(), # [eV/Bavg]
                'phot_N' : phot_bath_spec.N, # [ph/Bavg]
                'dE_inj_per_B' : dE_inj_per_Bavg, # [eV/Bavg]
                'dE_inj_per_Bavg_unclustered' : dE_inj_per_Bavg_unclustered, # [eV/Bavg]
                'dep_ion'  : np.mean(tf_wrapper.dep_box[...,0] + tf_wrapper.dep_box[...,1]), # [eV/Bavg]
                'dep_exc'  : np.mean(tf_wrapper.dep_box[...,2]), # [eV/Bavg]
                'dep_heat' : np.mean(tf_wrapper.dep_box[...,3]), # [eV/Bavg]
                'x_e_slice' : np.array(spin_temp.x_e_box[0]), # [1]
                'x_H_slice' : np.array(ionized_box.xH_box[0]), # [1]
                'T_k_slice' : np.array(spin_temp.Tk_box[0]), # [K]
            })

        if not use_tqdm:
            print(print_str, flush=True)

    #===== end of loop, return results =====
    arr_records = {k: np.array([r[k] for r in records]) for k in records[0].keys()}
    profiler.print_summary()

    return_dict = {
	    'profiler' : profiler,
        'records' : arr_records,
        'brightness_temp' : brightness_temp,
    }

    return return_dict

#===== utilities for evolve =====

def get_z_edges(z_max, z_min, zplusone_step_factor):
    z_s = [z_min]
    while z_s[-1] < z_max:
        z_s.append((z_s[-1] + 1.) * zplusone_step_factor - 1.)

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


def geom_inds(i_max, i_transition, n_goal):
    """Return a geometrically spaced index array with a dense start.

    Args:
        i_max (int):        Maximum available index.
        i_transition (int): Index where the geometric spacing starts.
        n_goal (int):       Target number of indices in the output array (actual number may vary slightly).

    Returns:
        np.array: Geometrically spaced index array.
    """
    if n_goal >= i_max:
        return np.arange(i_max)
    if n_goal <= i_transition:
        return np.arange(n_goal)
    # after this, i_transition < n_goal < i_max
    dense_arr = np.arange(i_transition)
    geom_arr = np.unique(np.round(np.geomspace(i_transition, i_max, n_goal-i_transition)).astype(int))
    return np.concatenate([dense_arr, geom_arr])