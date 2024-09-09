"""Main evolution function."""

import os
import sys
import logging
import gc

import numpy as np
from jax import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp

import py21cmfast as p21c
from py21cmfast import cache_tools

sys.path.append(os.environ['DH_DIR'])
from darkhistory.spec.spectrum import Spectrum

sys.path.append(os.environ['DM21CM_DIR'])
import dm21cm.physics as phys
from dm21cm.dh_wrapper import DarkHistoryWrapper
from dm21cm.tf_wrapper import TransferFunctionWrapper
from dm21cm.utils import load_h5_dict, init_logger
from dm21cm.xray_cache import XrayCache
from dm21cm.profiler import Profiler
from dm21cm.injections.zero import ZeroInjection

logging.getLogger('21cmFAST').setLevel(logging.CRITICAL)
logging.getLogger('py21cmfast._utils').setLevel(logging.CRITICAL)
logging.getLogger('py21cmfast.wrapper').setLevel(logging.CRITICAL)

logger = init_logger(__name__)


def evolve(run_name,
           z_start=None,
           z_end=None,
           subcycle_factor=10,
           max_n_shell=None,
           resume=False,
           use_tqdm=True,

           injection=None,
           p21c_initial_conditions=None,
           p21c_astro_params=None,

           use_DH_init=True,
           rerun_DH=False,
           homogenize_injection=False,
           homogenize_deposition=False,
           ):
    """
    Main evolution function.

    Args:
        run_name (str):               Name of run. Used for cache directory.
        z_start (float):              Starting redshift.
        z_end (float):                Ending redshift.
        subcycle_factor (int):        Number of DM21cm subcycles per 21cmFAST step.
        max_n_shell (int or None):    Max number total shells used in xray injection. If None, use all shells smaller than the box size.
        resume (bool):                Whether to attempt to resume from a previous run. Requires specifying the correct cache directory via run_name.
        use_tqdm (bool):              Whether to use tqdm progress bars.

        injection (Injection):        Injection object. If None, no injection is performed.
        p21c_initial_conditions (p21c.InitialConditions):  Initial conditions for 21cmFAST.
        p21c_astro_params (p21c.AstroParams):              AstroParams for 21cmFAST.
        
        use_DH_init (bool):           Whether to use DarkHistory initial conditions.
        rerun_DH (bool):              Whether to rerun DarkHistory to get initial values.
        homogenize_injection (bool):  Whether to use homogeneous injection, where DM density is averaged over the box.
        homogenize_deposition (bool): Whether to use homogeneous deposition, where the transfer function input parameters
                                      T_k, x_e, and delta are averaged over the box.

    Returns:
        dict: Dictionary of results consisting of:
            'global' (dict): Records of global quantities.
            'lightcone' (p21c.LightCone): Lightcones of density, x_e, T_k, T_s, and T_b.
            'profiler' (Profiler): Profiler.
    """

    logger.info(f'Using 21cmFAST version {p21c.__version__}')

    #===== data and cache =====
    data_dir = os.environ['DM21CM_DATA_DIR']
    cache_dir = os.environ['P21C_CACHE_DIR'] + '/' + run_name
    p21c.config['direc'] = cache_dir
    logger.info(f"Cache dir: {cache_dir}")
    os.makedirs(cache_dir, exist_ok=True)
    if not resume:
        cache_tools.clear_cache()
    gc.collect()

    #===== initialize =====
    #--- physics parameters ---
    abscs = load_h5_dict(f"{data_dir}/abscissas.h5")

    EPSILON = 1e-6
    p21c.global_params.Z_HEAT_MAX = z_start# + EPSILON
    p21c.global_params.ZPRIME_STEP_FACTOR = abscs['zplusone_step_factor'] ** subcycle_factor

    box_dim = p21c_initial_conditions.user_params.HII_DIM
    box_len = p21c_initial_conditions.user_params.BOX_LEN

    if injection:
        injection.set_binning(abscs)
        if injection.mode == 'DM p-wave annihilation' and injection.big_halo_contribution:
            logger.warning('This run cannot be resumed.')
            injection.init_delta_cache(data_dir=cache_dir, box_dim=box_dim, dx=box_len/box_dim)

        tfs = TransferFunctionWrapper(
            box_dim = box_dim,
            abscs = abscs,
            prefix = data_dir,
            enable_elec = injection.is_injecting_elec(),
            on_device = True,
        )
        if resume:
            xray_cache = XrayCache(data_dir=cache_dir, box_dim=box_dim, dx=box_len/box_dim, load_snapshot=True)
        else:
            xray_cache = XrayCache(data_dir=cache_dir, box_dim=box_dim, dx=box_len/box_dim)
            xray_cache.clear_cache()

    #===== initial steps =====
    # We synchronize DM21cm with 21cmFAST at the second step because 21cmFAST acts strangely in the first step:
    # - global_params.TK_at_Z_HEAT_MAX is not set correctly (it is likely set and evolved for a step).
    # - global_params.XION_at_Z_HEAT_MAX is not set correctly (it is likely set and evolved for a step).
    # - first step ignores any values added to spin_temp.Tk_box and spin_temp.x_e_box.

    z_edges_coarse = get_z_edges(z_start, z_end, p21c.global_params.ZPRIME_STEP_FACTOR) # steps for 21cmFAST
    z_edges = get_z_edges(z_edges_coarse[0]+EPSILON, z_end, abscs['zplusone_step_factor'])[1:] # steps for DM21cm, with the same first and last redshifts as z_edges_coarse

    z_edges_coarse = np.around(z_edges_coarse, decimals = 10) # roundoff to avoid floating point issues, the same is done in 21cmFAST
    z_edges = np.around(z_edges, decimals = 10)
    scrollz = z_edges_coarse.copy() # used in lightcone construction

    # Construct the initial state, which will be above Z_HEAT_MAX
    perturbed_field = p21c.perturb_field(redshift=z_edges_coarse[0], init_boxes=p21c_initial_conditions, write=True)
    spin_temp, ionized_box, brightness_temp = p21c_step(perturbed_field=perturbed_field, spin_temp=None, ionized_box=None, astro_params=p21c_astro_params)

    # Step past Z_HEAT_MAX to the synchronization point. Remove the initial z_edges from the list as well.
    z_edges_coarse = z_edges_coarse[1:]
    z_edges = z_edges[subcycle_factor:]
    z_match = z_edges_coarse[0]

    perturbed_field = p21c.perturb_field(redshift=z_match, init_boxes=p21c_initial_conditions, write=True)
    spin_temp, ionized_box, brightness_temp = p21c_step(perturbed_field=perturbed_field, spin_temp=spin_temp, ionized_box=ionized_box, astro_params=p21c_astro_params)

    if use_DH_init: # still can use DH to get initial conditions if no_injection is set
        dh_injection = ZeroInjection() if injection is None else injection
        dh_injection.set_binning(abscs)
        dh = DarkHistoryWrapper(dh_injection, prefix=p21c.config[f'direc'])
        dh.evolve(end_rs=(1+z_match)*0.9, rerun=rerun_DH)
        T_k_DH_init, x_e_DH_init, phot_bath_spec = dh.get_init_cond(rs=1+z_match)
        spin_temp.Tk_box += T_k_DH_init - np.mean(spin_temp.Tk_box)
        spin_temp.x_e_box += x_e_DH_init - np.mean(spin_temp.x_e_box)
        ionized_box.xH_box = 1 - spin_temp.x_e_box
    else:
        phot_bath_spec = Spectrum(abscs['photE'], np.zeros_like(abscs['photE']), spec_type='N', rs=1+z_match) # [ph / Bavg]
    if injection:
        if xray_cache.isresumed:
            phot_bath_spec = xray_cache.saved_phot_bath_spec


    #===== main loop =====
    i_z_range = range(len(z_edges)-1) # -1 such that the z_next in the final step will be z_end
    if use_tqdm:
        from tqdm import tqdm
        i_z_range = tqdm(i_z_range)

    #--- trackers ---
    records = []
    profiler = Profiler()

    #--- loop ---
    for i_z in i_z_range:

        profiler.start()
        i_z_coarse = i_z // subcycle_factor

        #===== physical quantities =====
        z_current = z_edges[i_z]
        z_next = z_edges[i_z+1]
        dt = phys.dt_step(z_current, abscs['zplusone_step_factor'])

        delta_plus_one_box = 1 + np.asarray(perturbed_field.density)
        x_e_box = np.asarray(1 - ionized_box.xH_box)
        T_k_box = np.asarray(spin_temp.Tk_box)
        if injection:
            tfs.set_params(
                rs = 1+z_current,
                delta_plus_one_box = delta_plus_one_box,
                x_e_box = x_e_box,
                T_k_box = T_k_box,
                homogenize_deposition = homogenize_deposition
            )
            tfs.reset_phot() # reset photon each subcycle, but deposition is reset only after populating boxes
            tfs.increase_dt(dt) # increase deposition dt each subcycle
        
        #===== photon injection and energy deposition =====
        if injection and xray_cache.is_latest_z(z_current):
            
            #--- xray ---
            # First we dump to bath all cached states whose shell is larger than the box size.
            for state in xray_cache.states:
                if state.isinbath:
                    continue
                if phys.conformal_dx_between_z(z_current, state.z_end) > box_len/2:
                    phot_bath_spec += state.spectrum
                    state.isinbath = True
                else:
                    break

            # Then we select the chosen shell indices for deposition
            if max_n_shell is not None:
                i_max = i_z - xray_cache.i_shell_start
                inds_increasing = geom_inds(i_max=i_max, i_transition=10, n_goal=max_n_shell)
                inds_chosen_shells = i_z - inds_increasing
            else:
                inds_chosen_shells = list(range(xray_cache.i_shell_start, i_z)) # all shells smaller than the box size are chosen

            # Finally, we accumulate spectra from the non-chosen shells and deposit only on the chosen shells
            accumulated_shell_spec = Spectrum(abscs['photE'], np.zeros_like(abscs['photE']), spec_type='N', rs=1+z_current) # [ph/Bavg]

            for i_state, state in enumerate(xray_cache.states):
                if state.isinbath:
                    continue # skip states that are already in bath
                if i_state not in inds_chosen_shells:
                    accumulated_shell_spec += state.spectrum
                    continue

                smoothed_rel_eng_box = xray_cache.get_smoothed_box(state, z_current)
                xray_spec = state.spectrum + accumulated_shell_spec
                tfs.inject_phot(xray_spec, inject_type='xray', weight_box=smoothed_rel_eng_box)
                accumulated_shell_spec *= 0.

            profiler.record('xray')

            #--- bath and homogeneous portion of xray ---
            tfs.inject_phot(phot_bath_spec, inject_type='bath')

            #--- injection (on-the-spot) ---
            n_Bavg = phys.n_B * (1 + z_current)**3 # [Bavg / pcm^3]

            inj_rate_spec, weight_box = injection.inj_phot_spec_box(z_current, delta_plus_one_box=delta_plus_one_box)
            if homogenize_injection:
                weight_box = jnp.full_like(weight_box, jnp.mean(weight_box))
            tfs.inject_phot(inj_rate_spec * dt / n_Bavg, weight_box=weight_box, inject_type='ots') # ingoing spec has [phot / Bavg]

            if injection.is_injecting_elec():
                inj_rate_spec, weight_box = injection.inj_elec_spec_box(z_current, delta_plus_one_box=delta_plus_one_box)
                if homogenize_injection:
                    weight_box = jnp.full_like(weight_box, jnp.mean(weight_box))
                tfs.inject_elec(inj_rate_spec * dt / n_Bavg, weight_box=weight_box)

            profiler.record('bath+ots')

            #===== prepare spectra for next step =====
            #--- bath (separating out xray) ---
            prop_phot_N = np.array(tfs.prop_phot_N) # propagating and emitted photons have been stored in tfs up to this point, time to get them out
            emit_phot_N = np.array(tfs.emit_phot_N)
            emit_bath_N, emit_xray_N = split_xray(emit_phot_N, abscs['photE'])
            phot_bath_spec = Spectrum(abscs['photE'], prop_phot_N + emit_bath_N, rs=1+z_current, spec_type='N') # photons not emitted to the xray band are added to the bath (treated as uniform)
            phot_bath_spec.redshift(1+z_next)

            #--- xray ---
            attenuation_arr = np.array(tfs.attenuation_arr(rs=1+z_current, x=1-np.mean(ionized_box.xH_box))) # convert from jax array
            xray_cache.advance_spectra(attenuation_arr, z_next)

            xray_spec = Spectrum(abscs['photE'], emit_xray_N, rs=1+z_current, spec_type='N') # [ph/Bavg]
            xray_spec.redshift(1+z_next)
            if np.mean(tfs.xray_eng_box) != 0.:
                # dont' normalize w.r.t. to np.dot(abscs['photE'], emit_xray_N) because
                # that contains not only the emission but propagation
                xray_rel_eng_box = tfs.xray_eng_box / jnp.mean(tfs.xray_eng_box) # [1 (relative energy)/Bavg]
            else:
                xray_rel_eng_box = np.zeros_like(tfs.xray_eng_box) # [1 (relative energy)/Bavg]
            xray_cache.cache(z_current, z_next, xray_spec, xray_rel_eng_box)

            #--- p-wave delta cache ---
            if injection.mode == 'DM p-wave annihilation' and injection.big_halo_contribution:
                injection.delta_cache.cache(z_current, z_next, delta_plus_one_box)

            profiler.record('prep next')

        #===== 21cmFAST step =====
        # check if z_next matches
        if (i_z_coarse + 1) * subcycle_factor == (i_z + 1):
            assert np.isclose(z_next, z_edges_coarse[i_z_coarse+1]) # cross check redshifts match on cycle

            perturbed_field = p21c.perturb_field(redshift=z_edges_coarse[i_z_coarse+1], init_boxes=p21c_initial_conditions)
            input_heating, input_ionization, input_jalpha = init_input_boxes(z_next, p21c_initial_conditions)
            if injection:
                tfs.populate_injection_boxes(input_heating, input_ionization, input_jalpha)
            spin_temp, ionized_box, brightness_temp = p21c_step(
                perturbed_field, spin_temp, ionized_box,
                input_heating = input_heating,
                input_ionization = input_ionization,
                input_jalpha = input_jalpha,
                astro_params = p21c_astro_params
            )
            if injection:
                xray_cache.save_snapshot(phot_bath_spec=phot_bath_spec) # only save snapshot once dep_box is cleared

            profiler.record('21cmFAST')

            #===== calculate and save some quantities =====
            records.append({
                'z'   : z_next,
                'T_s' : np.mean(spin_temp.Ts_box), # [K]
                'T_b' : np.mean(brightness_temp.brightness_temp), # [mK]
                'T_k' : np.mean(spin_temp.Tk_box), # [K]
                'x_e' : np.mean(spin_temp.x_e_box), # [1]
                '1-x_H' : np.mean(1 - ionized_box.xH_box), # [1]
            })
            if injection:
                records[-1].update({
                    'phot_N' : phot_bath_spec.N, # [ph/Bavg]
                    'inj_E_per_Bavg' : injection.inj_power(z_current) * dt / n_Bavg, # [eV/Bavg]
                    'dep_ion'  : np.mean(tfs.dep_box[...,0] + tfs.dep_box[...,1]), # [eV/Bavg]
                    'dep_exc'  : np.mean(tfs.dep_box[...,2]), # [eV/Bavg]
                    'dep_heat' : np.mean(tfs.dep_box[...,3]), # [eV/Bavg]
                })
    #===== end of loop =====

    #===== construct lightcone =====
    lightcone = p21c.run_lightcone(
        redshift = brightness_temp.redshift,
        user_params = brightness_temp.user_params,
        cosmo_params = brightness_temp.cosmo_params,
        astro_params = brightness_temp.astro_params,
        flag_options = brightness_temp.flag_options,
        lightcone_quantities = ['brightness_temp','Ts_box', 'Tk_box', 'x_e_box', 'xH_box', 'density'],
        scrollz = scrollz,
    )
    lightcone._write(fname=f'lightcones.h5', direc=cache_dir, clobber=True)

    profiler.record('lightcone')

    #===== return results =====
    global_records = {k: np.array([r[k] for r in records]) for k in records[0].keys()}
    np.save(f'{cache_dir}/global_records', global_records)

    profiler.print_summary()

    return {
        'global' : global_records,
        'lightcone' : lightcone,
        'profiler' : profiler,
    }


#===== utilities for evolve =====
def get_z_edges(z_max, z_min, zplusone_step_factor):
    z_s = [z_min]
    while z_s[-1] < z_max:
        z_s.append((z_s[-1] + 1.) * zplusone_step_factor - 1.)
    return z_s[::-1]


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


def init_input_boxes(z_next, p21c_initial_conditions):

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
