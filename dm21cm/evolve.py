import os
import sys
import time
import pickle
import h5py
import logging
import warnings
from tqdm import tqdm

import numpy as np
import jax
import jax.numpy as jnp

import py21cmfast as p21c
from   py21cmfast import cache_tools

sys.path.append(os.environ['DH_DIR'])
from darkhistory.spec.spectrum import Spectrum # use branch test_dm21cm

sys.path.append("..")

from   dm21cm.data_loader import load_data
from   dm21cm.field_smoother import WindowedData
import dm21cm.physics as phys
from   dm21cm.utils import load_dict

logging.getLogger().setLevel(logging.INFO)
logging.getLogger('21cmFAST').setLevel(logging.CRITICAL+1)
logging.getLogger('py21cmfast._utils').setLevel(logging.CRITICAL+1)
logging.getLogger('py21cmfast.wrapper').setLevel(logging.CRITICAL+1)
logging.info(f'Using 21cmFAST version {p21c.__version__}')


#========== Main evolve function ==========

def evolve(run_name, run_mode='xray',
           z_start=..., z_end=..., zplusone_step_factor=1.05,
           dm_params=..., struct_boost_model='erfc 1e-3', enable_elec=False,
           dhinit_list=['phot', 'T_k', 'x_e'], dhtf_version=...,
           p21c_initial_conditions=...,
           rerun_DH=False, clear_cache=False, force_reload_tf=False,
           use_tqdm=True, save_slices=True):
    """Main evolution function.

    Args:
        run_name (str)
        run_mode {'no inj', 'bath', 'xray'} : Injection mode.
        
        z_start (float)
        z_end (float)
        zplusone_step_factor (float)
        dm_params (DMParams)
        struct_boost_model {'erfc 1e-3', 'erfc 1e-6', 'erfc 1e-9'} : Structure boost model for annihilation.
        enable_elec (bool) : Whether to enable electron processes.
        dhinit_list (list) : List of variables to initialize at start with a DarkHistory run. Can include any of 'phot',
            'T_k', 'x_e'.
        dhtf_version (str) : Version of DarkHistory transfer function to use.
        p21c_initial_conditions (InitialConditions)
        
        rerun_DH (bool) : Whether to rerun DarkHistory to get initial values.
        clear_cache (bool) : Whether to clear cache for 21cmFAST.
        force_reload_tf (bool): Whether to force reload transfer functions. Use when changing dhtf_version.
        
        use_tqdm (bool)
        save_slices (bool) : Whether to save slices of T_b, T_k, ... of run.
    """
    
    #===== Cache =====
    p21c.config['direc'] = os.environ['P21C_CACHE_DIR'] + '/' + run_name
    os.makedirs(p21c.config['direc'], exist_ok=True)
    if clear_cache:
        cache_tools.clear_cache()
    
    #===== Initialize physics parameters =====
    p21c.global_params.Z_HEAT_MAX = z_start
    p21c.global_params.ZPRIME_STEP_FACTOR = zplusone_step_factor
    z_edges = get_z_edges(z_start, z_end, zplusone_step_factor)
    z_mids = np.sqrt(z_edges[1:] * z_edges[:-1])
    z_dh_stops = z_edges[1]
    
    abscs = load_dict(f"../data/abscissas/abscs_{dhtf_version}.h5")
    if not np.isclose(np.log(zplusone_step_factor), abscs['dlnz']):
        raise ValueError('zplusone_step_factor and dhtf_version mismatch')
    photeng = abscs['photE']
    eleceng = abscs['elecEk']
    
    p21c.global_params.CLUMPING_FACTOR = 1.
    box_dim = p21c_initial_conditions.user_params.HII_DIM
    box_len = p21c_initial_conditions.user_params.BOX_LEN
    
    if run_mode == 'xray':
        from astropy.cosmology import Planck18
        
        ex_lo, ex_hi = 1e2, 1e4 # [eV]
        ix_lo = np.searchsorted(photeng, ex_lo) # i of first bin greater than ex_lo, excluded
        ix_hi = np.searchsorted(photeng, ex_hi) # i of first bin greater than ex_hi, included
        attenuation_arr = np.ones((len(z_mids), len(photeng))) # same redshift locations as z_mids
        xray_shell_Rmax = box_len # [p-Mpc]
        
        xray_fn = p21c.config['direc']+'/xray_brightness.h5'
        try:
            os.remove(xray_fn)
            logging.info('xray file removed')
        except:
            logging.info('xray file not found')
        xray_windowed_data = WindowedData(
            data_path = xray_fn,
            cosmo = Planck18,
            N = box_dim,
            dx = box_len / box_dim,
            cache = True,
        )
        
    #===== Run DarkHistory =====
    if len(dhinit_list) > 0:
        dhinit_fn = f"{p21c.config['direc']}/dhinit_soln.p"

        if os.path.exists(dhinit_fn) and not rerun_DH:
            dhinit_soln = pickle.load(open(dhinit_fn, 'rb'))
        else:
            logging.info('Running DarkHistory to generate initial conditions.')

            import main
            dhinit_soln = main.evolve(
                DM_process=dm_params.mode, mDM=dm_params.m_DM,
                sigmav=dm_params.sigmav, primary=dm_params.primary,
                struct_boost=phys.struct_boost_func(model=struct_boost_model),
                start_rs=3000, end_rs=(1+z_dh_stops)*0.9, coarsen_factor=12, verbose=1,
                reion_switch=False
            )
            pickle.dump(dhinit_soln, open(dhinit_fn, 'wb'))
        
    #===== Initialize transfer functions =====
    if run_mode != 'no inj':
        data_prefix = os.environ['DM21CM_DATA_DIR'] + f'/tf/{dhtf_version}/phot'
        phot_prop_tf = load_data('phot_prop', prefix=data_prefix, reload=force_reload_tf)
        phot_scat_tf = load_data('phot_scat', prefix=data_prefix, reload=force_reload_tf)
        phot_dep_tf = load_data('phot_dep', prefix=data_prefix, reload=force_reload_tf)
        phot_prop_tf.set_fixed_in_spec(dm_params.inj_phot_spec.N)
        phot_scat_tf.set_fixed_in_spec(dm_params.inj_phot_spec.N)
        phot_dep_tf.set_fixed_in_spec(dm_params.inj_phot_spec.N)
        
        if enable_elec:
            raise NotImplementedError
            elec_phot_tf = load_data('elec_phot', prefix=data_prefix, reload=force_reload_tf)
            elec_dep_tf = load_data('elec_dep', prefix=data_prefix, reload=force_reload_tf)
            elec_phot_tf.set_fixed_in_spec(dm_params.inj_elec_spec.N)
            elec_dep_tf.set_fixed_in_spec(dm_params.inj_elec_spec.N)
        
        
    #===== LOOP LOOP LOOP LOOP =====
    records = []
    input_time_tot = 0.
    p21c_time_tot = 0.
    
    if save_slices:
        saved_slices = []
        i_slice = int(box_dim/2)
    if use_tqdm:
        pbar = tqdm(total=len(z_edges)-1, position=0)
        
    for i_z in range(len(z_edges)-1):

        if not use_tqdm:
            print(f'i_z={i_z}/{len(z_edges)-1}', flush=True)
        input_timer = time.time()

        z = z_edges[i_z]
        prop_phot_N = np.zeros_like(photeng) # [N / Bavg]
        emit_phot_N = np.zeros_like(photeng) # [N / Bavg]
        dep_box = np.zeros((box_dim, box_dim, box_dim, len(abscs['dep_c'],)))
        # last dimension: ('H ion', 'He ion', 'exc', 'heat', 'cont', 'xray')
        input_heating = input_ionization = input_jalpha = None

        
        if i_z == 0: # At this step we will arrive at z_edges[0], so z_mid is not defined yet.
            spin_temp = None
            
        elif run_mode == 'no inj':
            if i_z == 1:
                logging.warning('Not injecting anything in this run!')

        else: # input from second step
            z_mid = z_mids[i_z-1] # At this step we will arrive at z_edges[i], passing through z_mids[i-1].

            input_heating = p21c.input_heating(redshift=z, init_boxes=p21c_initial_conditions, write=False)
            input_ionization = p21c.input_ionization(redshift=z, init_boxes=p21c_initial_conditions, write=False)
            input_jalpha = p21c.input_jalpha(redshift=z, init_boxes=p21c_initial_conditions, write=False)

            z_prev = z_edges[i_z-1]
            dt = phys.dt_between_z(z_prev, z) # [s]
            if dm_params.mode == 'swave':
                struct_boost = phys.struct_boost_func(model=struct_boost_model)(1+z_mid)
            else:
                struct_boost = 1
            n_Bavg = phys.n_B * (1+z_mid)**3 # [Bavg cm^-3]

            #----- boxes -----
            delta_box = jnp.asarray(perturbed_field.density)
            B_per_Bavg = 1 + delta_box
            rho_DM_box = (1 + delta_box) * phys.rho_DM * (1+z_mid)**3 # [eV cm^-3]
            x_e_box = jnp.asarray(1 - ionized_box.xH_box)
            inj_per_Bavg_box = phys.inj_rate(rho_DM_box, dm_params) * dt * struct_boost / n_Bavg # [inj/Bavg]

            tf_kwargs = dict(
                rs = 1 + z_mid,
                nBs_s = (1+delta_box).ravel(),
                x_s = x_e_box.ravel(),
                out_of_bounds_action = 'clip',
            )

            #----- attenuation -----
            if run_mode == 'xray':
                dep_tf_at_point = phot_dep_tf.point_interp(rs=1+z, nBs=1, x=np.mean(x_e_box))
                dep_toteng = np.sum(dep_tf_at_point[:, :4], axis=1)
                attenuation_arr[i_z-1, :] = 1 - dep_toteng/photeng

            #===== photon bath -> prop emit dep =====
            prop_phot_N += phot_prop_tf(
                in_spec=phot_bath_spec.N, sum_result=True, **tf_kwargs,
            ) / (box_dim ** 3) # [N / Bavg]

            emit_phot_N += phot_scat_tf(
                in_spec=phot_bath_spec.N, sum_result=True, **tf_kwargs,
            ) / (box_dim ** 3) # [N / Bavg]

            dep_box += phot_dep_tf(
                in_spec=phot_bath_spec.N, sum_result=False, **tf_kwargs,
            ).reshape(dep_box.shape) # [eV / Bavg]

            #===== DM (prompt phot+elec) -> emit dep =====
            emit_phot_N += phot_prop_tf(
                in_spec=dm_params.inj_phot_spec.N, sum_result=True, sum_weight=inj_per_Bavg_box.ravel(), **tf_kwargs,
            ) / (box_dim ** 3) # [N / Bavg]

            emit_phot_N += phot_scat_tf(
                in_spec=dm_params.inj_phot_spec.N, sum_result=True, sum_weight=inj_per_Bavg_box.ravel(), **tf_kwargs,
            ) / (box_dim ** 3) # [N / Bavg]

            dep_box += phot_dep_tf(
                in_spec=dm_params.inj_phot_spec.N, sum_result=False, **tf_kwargs,
            ).reshape(dep_box.shape) * inj_per_Bavg_box[..., None] # [eV / Bavg]

            if enable_elec:
                emit_phot_N += elec_phot_tf(
                    in_spec=dm_params.inj_elec_spec.N, sum_result=True, sum_weight=inj_per_Bavg_box.ravel(), **tf_kwargs,
                ) / (box_dim ** 3) # [N / Bavg]

                dep_box += elec_dep_tf(
                    in_spec=dm_params.inj_elec_spec.N, sum_result=False, **tf_kwargs,
                ).reshape(dep_box.shape) * inj_per_Bavg_box[..., None] # [eV / Bavg]
                
            #===== emitted xray -> emit dep =====
            if run_mode == 'xray':

                xray_unif_N = np.zeros_like(photeng)

                for i_z_shell in range(2, i_z):
                    R1, R2 = xray_windowed_data.get_smoothing_radii(
                        z, z_edges[i_z_shell], z_edges[i_z_shell+1]
                    )
                    if not use_tqdm:
                        print(f'  i_z_shell={i_z_shell} {R2:.1f} -> {R1:.1f} Mpc', end='', flush=True)

                    if np.max([R1, R2]) < xray_shell_Rmax:
                        xray_e_box, xray_N = xray_windowed_data.get_smoothed_shell(
                            z_receiver = z,
                            z_donor = z_edges[i_z_shell],
                            z_next_donor = z_edges[i_z_shell+1]
                        )

                        xray_spec = Spectrum(photeng, xray_N, rs=z_edges[i_z_shell], spec_type='N') # [photon / Bavg]
                        for j in range(i_z_shell, i_z+1): # j is the i_z to get to.
                            xray_spec.N *= attenuation_arr[j-1]
                            xray_spec.redshift(1+z_edges[j])
                        # Note: xray_mean_eng = xray_spec.toteng() # unit: [eV / Bavg]
                        # Note: xray_e_box = local xray band energy / mean xray band energy
                        emit_phot_N += phot_scat_tf(
                            in_spec=xray_spec.N, sum_result=True, sum_weight=xray_e_box.ravel(), **tf_kwargs,
                        ) / (box_dim ** 3) # [N / Bavg]

                        dep_box += phot_dep_tf(
                            in_spec=xray_spec.N, sum_result=False, **tf_kwargs,
                        ).reshape(dep_box.shape) * xray_e_box[..., None] # [eV / Bavg] # CHECK SHAPE HERE

                    else:
                        if not use_tqdm:
                            print(f' exceeding Rmax={xray_shell_Rmax}. Going into uniform.', end='', flush=True)
                        xray_N = xray_windowed_data.get_spec(
                            z_receiver = z,
                            z_donor = z_edges[i_z_shell],
                            z_next_donor = z_edges[i_z_shell+1]
                        )

                        xray_spec = Spectrum(photeng, xray_N, rs=z_edges[i_z_shell], spec_type='N') # [photon / Bavg]
                        for j in range(i_z_shell, i_z+1): # j is the i_z to get to.
                            xray_spec.N *= attenuation_arr[j-1]
                            xray_spec.redshift(1+z_edges[j])

                        xray_unif_N += xray_spec.N

                    if not use_tqdm:
                        print('', flush=True)
                if not np.all(xray_unif_N == 0):
                    emit_phot_N += phot_scat_tf(
                        in_spec=xray_unif_N, sum_result=True, **tf_kwargs,
                    ) / (box_dim ** 3) # [N / Bavg]

                    dep_box += phot_dep_tf(
                        in_spec=xray_unif_N, sum_result=False, **tf_kwargs,
                    ).reshape(dep_box.shape) # [eV / Bavg] # CHECK SHAPE HERE
            else:
                pass # do nothing, because xray will be in photon bath already

            #===== update input_boxes =====
            input_heating.input_heating += np.array(
                2 / (3*phys.kB*(1+x_e_box)) * dep_box[...,3] / B_per_Bavg
            ) # [K/Bavg] / [B/Bavg] = [K/B]
            input_ionization.input_ionization += np.array(
                (dep_box[...,0] + dep_box[...,1]) / phys.rydberg / B_per_Bavg
            ) # [1/Bavg] / [B/Bavg] = [1/B]

            n_lya = dep_box[...,2] * n_Bavg / phys.lya_eng # [lya cm^-3]
            dnu_lya = (phys.rydberg - phys.lya_eng) / (2*np.pi*phys.hbar) # [Hz^-1]
            J_lya = n_lya * phys.c / (4*np.pi) / dnu_lya # [lya cm^-2 s^-1 sr^-1 Hz^-1]
            input_jalpha.input_jalpha += np.array(J_lya)

            #===== record =====
            dE_inj_per_Bavg = dm_params.eng_per_inj * np.mean(inj_per_Bavg_box) # [eV per Bavg]
            dE_inj_per_Bavg_unclustered = dE_inj_per_Bavg / struct_boost
            record_inj = {
                'dE_inj_per_B' : dE_inj_per_Bavg,
                'f_heat' : np.mean(dep_box[...,3]) / dE_inj_per_Bavg_unclustered,
                'f_ion'  : np.mean(dep_box[...,0] + dep_box[...,1]) / dE_inj_per_Bavg_unclustered,
                'f_exc'  : np.mean(dep_box[...,2]) / dE_inj_per_Bavg_unclustered,
            }

        input_time_tot += time.time() - input_timer

        #===== step in 21cmFAST =====
        p21c_timer = time.time()
        perturbed_field = p21c.perturb_field( # perturbed_field controls the redshift
            redshift=z,
            init_boxes=p21c_initial_conditions
        )
        spin_temp = p21c.spin_temperature(
            perturbed_field=perturbed_field,
            previous_spin_temp=spin_temp,
            input_heating_box=input_heating,
            input_ionization_box=input_ionization,
            input_jalpha_box=input_jalpha,
            write=True
        )
        ionized_box = p21c.ionize_box(
            spin_temp=spin_temp
        )
        brightness_temp = p21c.brightness_temperature(
            ionized_box=ionized_box,
            perturbed_field=perturbed_field,
            spin_temp=spin_temp
        )
        coeval = p21c.Coeval(
            redshift = z,
            initial_conditions = p21c_initial_conditions,
            perturbed_field = perturbed_field,
            ionized_box = ionized_box,
            brightness_temp = brightness_temp,
            ts_box = spin_temp,
        )
        p21c_time_tot += time.time() - p21c_timer

        if i_z == 0:
            z_now = z_edges[1] # matching after z_edges[0]->z_edges[1] step
            
            if 'T_k' in dhinit_list:
                T_k_DH = np.interp(z_now, dhinit_soln['rs'][::-1] - 1, dhinit_soln['Tm'][::-1] / phys.kB) # [K]
                spin_temp.Tk_box += T_k_DH - np.mean(spin_temp.Tk_box)
                

            if 'x_e' in dhinit_list:
                x_e_DH = np.interp(z_now, dhinit_soln['rs'][::-1] - 1, dhinit_soln['x'][::-1, 0]) # HI
                spin_temp.x_e_box += x_e_DH - np.mean(spin_temp.x_e_box)

            if 'phot' in dhinit_list:
                logrs_dh_arr = np.log(dhinit_soln['rs'])[::-1]
                logrs = np.log(1+z_now)
                i = np.searchsorted(logrs_dh_arr, logrs)
                logrs_left, logrs_right = logrs_dh_arr[i-1:i+1]
                
                dh_spec_N_arr = np.array([s.N for s in dhinit_soln['highengphot']])[::-1]
                dh_spec_left, dh_spec_right = dh_spec_N_arr[i-1:i+1]
                dh_spec = ( dh_spec_left * np.abs(logrs - logrs_right) + \
                            dh_spec_right * np.abs(logrs - logrs_left) ) / np.abs(logrs_right - logrs_left)
                phot_bath_spec = Spectrum(photeng, dh_spec, rs=1+z_now, spec_type='N')
            else:
                phot_bath_spec = Spectrum(photeng, np.zeros_like(photeng), rs=1+z_now, spec_type='N') # [N per Bavg]

        #===== prepare for next step =====
        if i_z >= 1:
            if run_mode == 'xray':
                emit_bath_N, emit_xray_N = split_xray(emit_phot_N, ix_lo, ix_hi)
                out_phot_N = prop_phot_N + emit_bath_N

                if not use_tqdm:
                    print(f'  xray band energy = {np.dot(photeng, emit_xray_N):.3e} = {np.mean(dep_box[..., 5]):.3e} eV / Bavg', flush=True)
                xray_e_box = dep_box[..., 5] / np.dot(photeng, emit_xray_N)
                # The mean energy for dep_box[..., 5] should be close to emit_xray_N.toteng()
                # for the above to work, every time emit_phot_N is populated, dep_box must be populated by the same input spectrum.
                xray_windowed_data.set_field(field=xray_e_box, spec=emit_xray_N, z=z)
                # xray_windowed_data.global_Tk = np.append(xray_windowed_data.global_Tk, np.mean(spin_temp.Tk_box))
                # xray_windowed_data.global_x = np.append(xray_windowed_data.global_x, np.mean(ionized_field.xH_box))
            else:
                out_phot_N = prop_phot_N + emit_phot_N # treat everything as uniform

            out_phot_spec = Spectrum(photeng, out_phot_N, rs=1+z, spec_type='N')    
            
            if z != z_edges[-1]:
                out_phot_spec.redshift(1+z_edges[i_z+1])
                phot_bath_spec = out_phot_spec

        #===== save results =====
        if i_z > 0:
            record = {
                'z'   : z_edges[i_z+1],
                'T_s' : np.mean(spin_temp.Ts_box), # [mK]
                'T_b' : np.mean(brightness_temp.brightness_temp), # [K]
                'T_k' : np.mean(spin_temp.Tk_box), # [K]
                'x_e' : np.mean(1 - ionized_box.xH_box), # [1]
            }
            if run_mode in ['bath', 'xray']:
                record.update(record_inj)
            records.append(record)

        if save_slices:
            saved_slices.append({
                'z'   : z,
                'T_s' : spin_temp.Ts_box[i_slice], # [mK]
                'T_b' : brightness_temp.brightness_temp[i_slice], # [K]
                'T_k' : spin_temp.Tk_box[i_slice], # [K]
                'x_e' : 1 - ionized_box.xH_box[i_slice], # [1]
                'delta' : perturbed_field.density[i_slice], # [1]
            })

        if use_tqdm:
            pbar.update()

    #===== LOOP END LOOP END =====
    
    arr_records = {k: np.array([r[k] for r in records]) for k in records[0].keys()}
    np.save(f'../data/run_info/{run_name}_records', arr_records)
    if save_slices:
        np.save(f'../data/run_info/{run_name}_slices', saved_slices)
        
    print(f'input used {input_time_tot:.4f} s')
    print(f'p21c used {p21c_time_tot:.4f} s')
    
    
    
#========== Helper functions ==========

def get_z_edges(z_start, z_end, zplusone_step_factor):
    """Standard redshift array for evolve."""

    z_arr = [z_end]
    while z_arr[-1] < z_start:
        z = (1 + z_arr[-1]) * zplusone_step_factor - 1
        z_arr.append(z)
        
    return np.array(z_arr[::-1][1:])

def split_xray(phot_N, ix_lo, ix_hi):
    """Split a photon spectrum (N in bin) into bath and xray band."""
    bath_N = np.array(phot_N).copy()
    xray_N = np.array(phot_N).copy()
    bath_N[ix_lo:ix_hi] *= 0
    xray_N[:ix_lo] *= 0
    xray_N[ix_hi:] *= 0
    return bath_N, xray_N