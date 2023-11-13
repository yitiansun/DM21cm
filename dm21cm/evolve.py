"""Main evolution function."""

import os
import sys
import logging
import gc

import numpy as np
from scipy import interpolate
from scipy import optimize
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
           use_xray_interp_shell=True,
           ):
    """
    Main evolution function.

    Args:
        run_name (str):      Name of run. Used for cache directory.
        z_start (float):     Starting redshift.
        z_end (float):       Ending redshift.
        dm_params (dm21cm.dm_params.DMParams):             Dark matter (DM) parameters.
        enable_elec (bool):                                Whether to enable electron injection.
        p21c_initial_conditions (p21c.InitialConditions):  Initial conditions for 21cmFAST.
        p21c_astro_params (p21c.AstroParams):              AstroParams for 21cmFAST.
        use_DH_init (bool):  Whether to use DarkHistory initial conditions.
        rerun_DH (bool):     Whether to rerun DarkHistory to get initial values.
        clear_cache (bool):  Whether to clear cache for 21cmFAST.
        use_tqdm (bool):     Whether to use tqdm progress bars.
        tf_on_device (bool): Whether to put transfer functions on device (GPU).
        no_injection (bool): Whether to skip injection and energy deposition.
        use_xray_interp_shell (bool): Whether to use interpolation shell for xray injection.

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
    p21c.global_params.ZPRIME_STEP_FACTOR = abscs['zplusone_step_factor']

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
    if use_xray_interp_shell:
        xray_cacher = Cacher(box_dim=box_dim, dx=box_len/box_dim)
    else:
        from dm21cm.data_cacher_old import Cacher
        xray_cacher = Cacher(data_path=f"{cache_dir}/xray_brightness.h5", cosmo=cosmo, N=box_dim, dx=box_len/box_dim)
        xray_cacher.clear_cache()

    #--- redshift stepping ---
    z_edges = get_z_edges(z_start, z_end, p21c.global_params.ZPRIME_STEP_FACTOR)

    #===== initial steps =====
    dh_wrapper = DarkHistoryWrapper(dm_params, prefix=p21c.config[f'direc'])

    # We have to synchronize at the second step because 21cmFAST acts weird in the first step:
    # - global_params.TK_at_Z_HEAT_MAX is not set correctly (it is probably set and evolved for a step)
    # - global_params.XION_at_Z_HEAT_MAX is not set correctly (it is probably set and evolved for a step)
    # - first step ignores any values added to spin_temp.Tk_box and spin_temp.x_e_box
    z_match = z_edges[1]
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
    z_edges = z_edges[1:] # Maybe fix this later
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

        print_str = f'i_z={i_z}/{len(z_edges)-2} z={z_edges[i_z]:.2f}'

        #===== physical quantities =====
        z_current = z_edges[i_z]
        z_next = z_edges[i_z+1]
        dt = phys.dt_step(z_current, np.exp(abscs['dlnz']))

        #--- for interpolation ---
        delta_plus_one_box = 1 + np.asarray(perturbed_field.density)
        x_e_box = np.asarray(1 - ionized_box.xH_box)
        T_k_box = np.asarray(spin_temp.Tk_box)
        tf_wrapper.init_step(rs=1+z_current, delta_plus_one_box=delta_plus_one_box, x_e_box=x_e_box, T_k_box=T_k_box)

        #--- for dark matter ---
        nBavg = phys.n_B * (1+z_current)**3 # [Bavg / (physical cm)^3]
        rho_DM_box = delta_plus_one_box * phys.rho_DM * (1+z_current)**3 # [eV/(physical cm)^3]
        inj_per_Bavg_box = phys.inj_rate(rho_DM_box, dm_params) * dt * dm_params.struct_boost(1+z_current) / nBavg # [inj/Bavg]

        #===== photon injection and energy deposition =====
        
        profiler.start()

        if not no_injection:

            #--- xray interpolating shell ---
            if use_xray_interp_shell:

                if not len(xray_cacher.states)==0:

                    r_from_z = np.vectorize(lambda z: phys.conformal_dx_between_z(z_current, z)) # conformal distance [cMpc] of z from current shell
                    z_interp_arr = np.geomspace(z_current, 1100., 1000) # up to CMB
                    r_interp_arr = r_from_z(z_interp_arr)
                    z_from_r = interpolate.interp1d(r_interp_arr, z_interp_arr, bounds_error=False, fill_value='extrapolate') # inverse of r_z
                    
                    r_shells = get_r_shells(box_dim, box_len, r_cap=r_from_z(np.max(xray_cacher.z_s)), n_target=40) # R_a in paper
                    z_shells = z_from_r(r_shells) # z_a in paper
                    z_shell_mids = np.concatenate([[z_current], (z_shells[:-1] + z_shells[1:]) / 2, [z_shells[-1]]])
                    r_shell_mids = r_from_z(z_shell_mids) # start and end for R windows
                    dz_shells = np.diff(z_shell_mids) # dz_a in paper

                    # Example (make a better comment later)
                    # inds         =   0, 1,   2,   ..., N-2, N-1    |
                    # r_shells     =   0, 1.2, 2.4, ..., 250, 256    | total=N
                    # r_shell_mids = 0, 0.6, 1.8, ..., 247, 253, 256 | total=N+1

                    for i, z_shell in enumerate(z_shells):

                        # z_left < z_shell < z_right
                        i_z_right = np.searchsorted(-z_edges, -z_shell, side='right')
                        i_z_left = i_z_right + 1
                        z_right = z_edges[i_z_right]
                        z_left = z_edges[i_z_left]

                        ftdEdz_right, rel_spec_right = xray_cacher.get_ftdEdz_spec(z_right)
                        ftdEdz_left,  rel_spec_left  = xray_cacher.get_ftdEdz_spec(z_left)
                        left_weight = (z_right - z_shell) / (z_right - z_left)
                        right_weight = 1 - left_weight

                        ftdEdz = left_weight * ftdEdz_left + right_weight * ftdEdz_right
                        dEdz, _ = xray_cacher.smooth_box(ftdEdz, r_shell_mids[i], r_shell_mids[i+1]) # r_shell_mids[i] < r_shell < r_shell_mids[i+1]
                        rel_spec = left_weight * rel_spec_left + right_weight * rel_spec_right

                        dE = dEdz * dz_shells[i] # [eV/Bavg]
                        tf_wrapper.inject_phot(rel_spec, inject_type='xray', weight_box=dE)

                    # We have summed all the shells up to r_shells[-1] (precisely), and we need to release the rest to bath
                    phot_bath_spec += xray_cacher.release_to_bath_prior_to(z_shells[-1])

            #--- xray (original) ---
            else:
                for i_z_shell in range(i_xray_loop_start, i_z):
                    xray_brightness_box, xray_spec, is_box_average = xray_cacher.get_annulus_data(
                        z_current, z_edges[i_z_shell], z_edges[i_z_shell+1]
                    )
                    if is_box_average:                                          # if smoothing scale > box size,
                        phot_bath_spec.N += xray_spec.N                         # then we can just dump to the global bath spectrum
                        i_xray_loop_start = max(i_z_shell+1, i_xray_loop_start) # and we will not revisit this shell
                    else:
                        tf_wrapper.inject_phot(xray_spec, inject_type='xray', weight_box=xray_brightness_box)

            profiler.record('xray')
            
            #--- bath and homogeneous portion of xray ---
            tf_wrapper.inject_phot(phot_bath_spec, inject_type='bath')

            #--- dark matter (on-the-spot) ---
            tf_wrapper.inject_from_dm(dm_params, inj_per_Bavg_box)

            profiler.record('bath+dm')

        #===== 21cmFAST step =====
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
        xray_cacher.advance_spectra(attenuation_arr, z_next)

        xray_spec = Spectrum(abscs['photE'], emit_xray_N, rs=1+z_current, spec_type='N') # [ph/Bavg]
        xray_spec.redshift(1+z_next)
        xray_tot_eng = np.dot(abscs['photE'], emit_xray_N)
        if xray_tot_eng == 0.:
            xray_rel_eng_box = np.zeros_like(tf_wrapper.xray_eng_box)
        else:
            xray_rel_eng_box = tf_wrapper.xray_eng_box / xray_tot_eng # [1 (relative energy)/Bavg]
        if not no_injection:
            xray_cacher.cache(z_current, xray_rel_eng_box, xray_spec)

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

        profiler.record('prep_next')

        if not use_tqdm:
            print(print_str, flush=True)

    #===== end of loop, return results =====
    arr_records = {k: np.array([r[k] for r in records]) for k in records[0].keys()}
    profiler.print_summary()

    res_summary = {
	    'profiler' : profiler,
        'records' : arr_records,
    }

    return brightness_temp, res_summary


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

def get_r_shells(box_dim, box_len, r_cap=None, n_target=40):
    """Generate r values for interpolation shells."""
    L_FACTOR = 0.620350491
    R = L_FACTOR * box_len/box_dim
    R_factor = (p21c.global_params.R_XLy_MAX/R) ** (1/p21c.global_params.NUM_FILTER_STEPS_FOR_Ts)
    r_s = R * R_factor**np.arange(n_target)
    r_s = np.append(r_s, p21c.global_params.R_XLy_MAX)
    r_s = np.insert(r_s, 0, 0.)
    if r_cap is not None:
        r_max = np.min([r_cap, box_len/2])
    else:
        r_max = box_len/2
    r_s = np.unique(np.minimum(r_s, r_max)) # smooth up to radii at half the box length
    return r_s