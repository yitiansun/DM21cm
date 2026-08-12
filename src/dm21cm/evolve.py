"""Main evolution function."""

import os
import sys
import shutil
import logging
import gc

import attrs
import numpy as np
from jax import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp

# $P21C_CACHE_DIR often lives on a parallel filesystem (e.g. Lustre), where HDF5's
# default flock-based locking blocks forever: the run wedges in state D inside
# flock() while writing the very first InitialConditions box. This must be set
# before the HDF5 library initializes, i.e. before py21cmfast/h5py are imported.
# setdefault, so an explicit setting from the environment still wins.
os.environ.setdefault('HDF5_USE_FILE_LOCKING', 'FALSE')

import py21cmfast as p21c
from py21cmfast import OutputCache, RunCache, RectilinearLightconer
from py21cmfast.drivers.lightcone import setup_lightcone_instance

from darkhistory.spec.spectrum import Spectrum

import dm21cm.physics as phys
from dm21cm.dh_wrapper import DarkHistoryWrapper
from dm21cm.tf_wrapper import TransferFunctionWrapper
from dm21cm.utils import load_h5_dict, init_logger, Profiler
from dm21cm.xray_cache import XrayCache
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
           use_tqdm=True,

           injection=None,
           p21c_initial_conditions=None,

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
        use_tqdm (bool):              Whether to use tqdm progress bars.

        injection (Injection):        Injection object. If None, no injection is performed.
        p21c_initial_conditions (p21c.InitialConditions):  Initial conditions for 21cmFAST. Its
                                      ``.inputs`` supplies every 21cmFAST parameter for the run
                                      (astro params and options included); only ``node_redshifts``
                                      is overridden here, since everything else is part of the
                                      initial conditions' compatibility hash. Build it with
                                      ``Z_HEAT_MAX=z_start``.

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
    logger.info(f"Cache dir: {cache_dir}")
    shutil.rmtree(cache_dir, ignore_errors=True) # start from a clean run cache
    os.makedirs(cache_dir, exist_ok=True)
    cache = OutputCache(cache_dir)
    gc.collect()

    #===== initialize =====
    #--- physics parameters ---
    abscs = load_h5_dict(f"{data_dir}/abscissas.h5")

    #--- redshift ladders ---
    # The coarse ladder is what 21cmFAST steps on, so it *is* node_redshifts. Note these
    # exact float objects have to be reused when asking for boxes: 21cmFAST matches a
    # `previous_*` box against its node redshift with `!=`, not a tolerance.
    z_edges, z_edges_coarse = get_z_edges(z_start, z_end, abscs['zplusone_step_factor'], subcycle_factor)

    #--- 21cmFAST parameters ---
    # Only node_redshifts is set here. Z_HEAT_MAX and ZPRIME_STEP_FACTOR live in
    # simulation_options, which is part of the InitialConditions compatibility hash, so
    # changing them would invalidate the initial conditions we were handed. They are the
    # caller's to set; we only check that Z_HEAT_MAX makes the first node the initializer.
    #
    # node_redshifts itself is *not* in that hash, but it is in the hash of every evolved
    # box, so it must be fixed once here and never re-cloned mid-run.
    z_heat_max = p21c_initial_conditions.inputs.simulation_options.Z_HEAT_MAX
    if not z_edges_coarse[1] < z_heat_max <= z_edges_coarse[0]:
        raise ValueError(
            f"Z_HEAT_MAX={z_heat_max} must lie in ({z_edges_coarse[1]}, {z_edges_coarse[0]}] so that "
            f"exactly the first coarse node is 21cmFAST's initializer box. Setting "
            f"Z_HEAT_MAX=z_start={z_start} when building the initial conditions always satisfies this."
        )

    inputs = p21c_initial_conditions.inputs.clone(node_redshifts=tuple(z_edges_coarse))

    # The caller may have built the initial conditions without writing them anywhere.
    # Put them in the run cache so it is self-contained, which is what lets the lightcone
    # be rebuilt from the cache afterwards.
    cache.write(p21c_initial_conditions)

    box_dim = inputs.simulation_options.HII_DIM
    box_len = inputs.simulation_options.BOX_LEN

    if injection:

        tfs = TransferFunctionWrapper(
            box_dim = box_dim,
            abscs = abscs,
            prefix = data_dir,
            enable_elec = injection.is_injecting_elec(),
            on_device = True,
        )
        xray_cache = XrayCache(data_dir=cache_dir, box_dim=box_dim, dx=box_len/box_dim)
        xray_cache.clear_cache()

    #===== initial step =====
    # The first box is the initializer: at z >= Z_HEAT_MAX 21cmFAST sets T_k and x_e
    # straight from RECFAST with no dzp step, so any injection handed to it is ignored.
    # DM21cm therefore hands over its own initial condition here, and starts injecting
    # from the next node.
    z_match = z_edges_coarse[0]

    perturbed_field = p21c_step_perturb(z_match, p21c_initial_conditions, inputs, cache)
    spin_temp, ionized_box, brightness_temp = p21c_step(
        perturbed_field, None, None, None, p21c_initial_conditions, inputs, cache,
    )

    if use_DH_init: # still can use DH to get initial conditions if no_injection is set
        dh_injection = ZeroInjection() if injection is None else injection
        dh = DarkHistoryWrapper(dh_injection, prefix=cache_dir)
        dh.evolve(end_rs=(1+z_match)*0.9, rerun=rerun_DH, start_rs=3000)
        T_k_DH_init, x_e_DH_init, phot_bath_spec = dh.get_init_cond(rs=1+z_match)
        T_k_box = np.asarray(spin_temp.get('kinetic_temp_neutral'))
        x_e_box = np.asarray(spin_temp.get('xray_ionised_fraction'))
        spin_temp.set('kinetic_temp_neutral', (T_k_box + T_k_DH_init - np.mean(T_k_box)).astype(np.float32))
        spin_temp.set('xray_ionised_fraction', (x_e_box + x_e_DH_init - np.mean(x_e_box)).astype(np.float32))
        ionized_box.set('neutral_fraction', (1 - np.asarray(spin_temp.get('xray_ionised_fraction'))).astype(np.float32))
    else:
        phot_bath_spec = Spectrum(abscs['photE'], np.zeros_like(abscs['photE']), spec_type='N', rs=1+z_match) # [ph / Bavg]


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

        delta_plus_one_box = 1 + np.asarray(perturbed_field.get('density'))
        x_e_box = np.asarray(1 - ionized_box.get('neutral_fraction'))
        T_k_box = np.asarray(spin_temp.get('kinetic_temp_neutral'))
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
        if injection:

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

            inj_rate_spec, weight_box = injection.inj_phot_spec_box(
                z_current,
                z_end = z_next,
                delta_plus_one_box = delta_plus_one_box,
                T_k_box = T_k_box,
                x_e_box = x_e_box,
            )
            if homogenize_injection:
                weight_box = jnp.full_like(weight_box, jnp.mean(weight_box))
            tfs.inject_phot(inj_rate_spec * dt / n_Bavg, weight_box=weight_box, inject_type='ots') # ingoing spec has [phot / Bavg]

            if injection.is_injecting_elec():
                inj_rate_spec, weight_box = injection.inj_elec_spec_box(
                    z_current,
                    z_end = z_next,
                    delta_plus_one_box = delta_plus_one_box,
                    T_k_box = T_k_box,
                    x_e_box = x_e_box,
                )
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
            attenuation_arr = np.array(tfs.attenuation_arr(rs=1+z_current, x=1-np.mean(ionized_box.get('neutral_fraction')))) # convert from jax array
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

            profiler.record('prep next')

        #===== 21cmFAST step =====
        # check if z_next matches
        if (i_z_coarse + 1) * subcycle_factor == (i_z + 1):
            previous_perturbed_field = perturbed_field # 21cmFAST needs it below Z_HEAT_MAX
            perturbed_field = p21c_step_perturb(
                z_edges_coarse[i_z_coarse+1], p21c_initial_conditions, inputs, cache,
            )
            # Snapshot the deposition means before get_injection_boxes() resets dep_box.
            dep_box_means = tfs.dep_box_means if injection else None
            injection_boxes = tfs.get_injection_boxes() if injection else (None, None, None)
            spin_temp, ionized_box, brightness_temp = p21c_step(
                perturbed_field, previous_perturbed_field, spin_temp, ionized_box,
                p21c_initial_conditions, inputs, cache,
                injection_boxes = injection_boxes,
            )

            profiler.record('21cmFAST')

            #===== calculate and save some quantities =====
            records.append({
                'z'   : z_next,
                'T_s' : np.mean(spin_temp.get('spin_temperature')), # [K]
                'T_b' : np.mean(brightness_temp.get('brightness_temp')), # [mK]
                'T_k' : np.mean(spin_temp.get('kinetic_temp_neutral')), # [K]
                'x_e' : np.mean(spin_temp.get('xray_ionised_fraction')), # [1]
                '1-x_H' : np.mean(1 - ionized_box.get('neutral_fraction')), # [1]
            })
            if injection:
                records[-1].update({
                    'phot_N' : phot_bath_spec.N, # [ph/Bavg]
                    'inj_E_per_Bavg' : injection.inj_power(z_current) * dt / n_Bavg, # [eV/Bavg]
                    'dep_ion'  : dep_box_means[0] + dep_box_means[1], # [eV/Bavg]
                    'dep_exc'  : dep_box_means[2], # [eV/Bavg]
                    'dep_heat' : dep_box_means[3], # [eV/Bavg]
                })
    #===== end of loop =====

    #===== construct lightcone =====
    lightcone = build_lightcone(inputs, cache, z_edges_coarse)
    lightcone.save(f'{cache_dir}/lightcones.h5', clobber=True)

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
def get_z_edges(z_max, z_min, zplusone_step_factor, subcycle_factor):
    """Fine (DM21cm) and coarse (21cmFAST) redshift ladders, both descending.

    Both are geometric in (1+z) anchored exactly at ``z_min``; the fine ladder steps by
    ``zplusone_step_factor`` and the coarse ladder is every ``subcycle_factor``-th fine
    node, so the coarse nodes *are* fine nodes rather than merely coinciding with them
    numerically. The number of fine steps is a multiple of ``subcycle_factor``, so both
    ladders share their first and last entries; the first overshoots ``z_max`` by less
    than one coarse step.

    Rounded to 10 decimals, as 21cmFAST does when it builds its own ladder.

    Args:
        z_max (float):                  Ladders extend to at least this redshift.
        z_min (float):                  Exact lowest redshift of both ladders.
        zplusone_step_factor (float):   (1+z)/(1+z_next) for the fine ladder.
        subcycle_factor (int):          Fine steps per coarse step.

    Returns:
        (array, array): the fine and coarse ladders, descending.
    """
    z_s = [z_min]
    while z_s[-1] < z_max or (len(z_s) - 1) % subcycle_factor != 0:
        z_s.append((z_s[-1] + 1.) * zplusone_step_factor - 1.)

    z_edges = np.around(z_s[::-1], decimals=10)

    return z_edges, z_edges[::subcycle_factor]


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


LIGHTCONE_QUANTITIES = (
    'brightness_temp', 'spin_temperature', 'kinetic_temp_neutral',
    'xray_ionised_fraction', 'neutral_fraction', 'density',
)


def build_lightcone(inputs, cache, z_edges_coarse, quantities=LIGHTCONE_QUANTITIES):
    """Assemble a lightcone from the coarse-step boxes left in the run cache.

    Done after the evolution rather than inside the loop, to keep the loop to the
    physics. The boxes are read back from the cache and interpolated pairwise by
    21cmFAST's own Lightconer, exactly as its internal lightcone driver does.

    We deliberately do *not* call ``p21c.run_lightcone``: 21cmFAST's cache key does not
    include the injection, so a driver that found any box missing would recompute it
    *without* injection and silently return a corrupted lightcone. Reading explicitly
    and refusing on an incomplete cache cannot do that.

    Args:
        inputs (p21c.InputParameters): The run's inputs; its node_redshifts are the ladder.
        cache (p21c.OutputCache):      Cache holding the run's boxes.
        z_edges_coarse (array):        The coarse (21cmFAST) ladder, descending.
        quantities (tuple):            Box fields to build lightcones of.

    Returns:
        p21c.LightCone
    """
    run_cache = RunCache.from_inputs(inputs, cache)
    if not run_cache.is_complete():
        missing = [str(run_cache.InitialConditions)] if not run_cache.InitialConditions.exists() else []
        for kind, paths in attrs.asdict(run_cache, recurse=False).items():
            if isinstance(paths, dict):
                missing += [f'{kind} at z={z}' for z, path in paths.items() if not path.exists()]
        raise RuntimeError(
            'The run cache is incomplete, refusing to build a lightcone: 21cmFAST would '
            'have to recompute the missing boxes, and since the injection is not part of '
            f'its cache key it would do so without injection. Missing: {missing}'
        )

    lightconer = RectilinearLightconer.between_redshifts(
        min_redshift = z_edges_coarse[-1],
        max_redshift = z_edges_coarse[0],
        resolution = inputs.simulation_options.cell_size,
        quantities = quantities,
    )
    lightcone = setup_lightcone_instance(
        lightconer = lightconer,
        scrollz = z_edges_coarse,
        inputs = inputs,
        include_dvdr_in_tau21 = False, # needs los_velocity; DM21cm does not ask for RSDs
        apply_rsds = False,
        photon_nonconservation_data = {},
    )

    previous_coeval = None
    for i_z in range(len(z_edges_coarse)):
        coeval = run_cache.get_coeval_at_z(index=i_z)

        # Mirrors 21cmFAST's own lightcone driver: the two turnover masses are box
        # attributes rather than grids, so they are not reachable via getattr(coeval, ...).
        for quantity in lightcone.global_quantities:
            if quantity == 'log10_mturn_acg':
                value = coeval.ionized_box.log10_Mturnover_ave
            elif quantity == 'log10_mturn_mcg':
                value = coeval.ionized_box.log10_Mturnover_MINI_ave
            else:
                value = np.mean(getattr(coeval, quantity))
            lightcone.global_quantities[quantity][i_z] = value

        if previous_coeval is not None:
            for quantity, idx, this_lc in lightconer.make_lightcone_slices(coeval, previous_coeval):
                if this_lc is not None:
                    lightcone.lightcones[quantity][..., idx] = this_lc

        previous_coeval = coeval

    return lightcone


def p21c_step_perturb(z, initial_conditions, inputs, cache):
    """Perturbed field at ``z``, written to the run cache."""
    return p21c.perturb_field(
        redshift = z,
        initial_conditions = initial_conditions,
        inputs = inputs,
        cache = cache,
        write = True,
    )


def p21c_step(perturbed_field, previous_perturbed_field, spin_temp, ionized_box,
              initial_conditions, inputs, cache, injection_boxes=(None, None, None)):
    """One 21cmFAST step, injecting the energy deposited since the previous step.

    ``injection_boxes`` is ``(heating, ionization, jalpha)``; heating and ionization are
    increments over this step (added by 21cmFAST alongside its own ``* dzp`` integration),
    jalpha is a flux. Any of them may be None, meaning no injection in that channel.

    Every box is written to the cache: the lightcone is assembled from it afterwards.
    """
    input_heating, input_ionization, input_jalpha = injection_boxes

    spin_temp = p21c.compute_spin_temperature(
        initial_conditions = initial_conditions,
        perturbed_field = perturbed_field,
        previous_spin_temp = spin_temp,
        inputs = inputs,
        input_heating = input_heating,
        input_ionization = input_ionization,
        input_jalpha = input_jalpha,
        cache = cache,
        write = True,
    )

    ionized_box = p21c.compute_ionization_field(
        initial_conditions = initial_conditions,
        perturbed_field = perturbed_field,
        previous_perturbed_field = previous_perturbed_field,
        previous_ionized_box = ionized_box,
        spin_temp = spin_temp,
        inputs = inputs,
        cache = cache,
        write = True,
    )

    brightness_temp = p21c.brightness_temperature( # takes no `inputs`: it reads them off the boxes
        ionized_box = ionized_box,
        perturbed_field = perturbed_field,
        spin_temp = spin_temp,
        cache = cache,
        write = True,
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
