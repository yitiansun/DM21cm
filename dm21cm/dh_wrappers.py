import os
import sys
import pickle
import logging
import numpy as np
from scipy import interpolate

sys.path.append("..")
import dm21cm.physics as phys
from dm21cm.interpolators_jax import BatchInterpolator

sys.path.append(os.environ['DH_DIR'])
from darkhistory.main import evolve as evolve_DH
from darkhistory.spec.spectrum import Spectrum


EPSILON = 1e-6


class DarkHistoryWrapper:
    
    def __init__(self, dm_params, prefix='.', soln_name='dh_init_soln.p'):

        self.dm_params = dm_params
        self.soln_fn = prefix + '/' + soln_name

    def clear_soln(self):
        if os.path.exists(self.soln_fn):
            logging.info('DarkHistoryWrapper: Removed cached DarkHistory run.')
            os.remove(self.soln_fn)

    def evolve(self, end_rs, rerun=False, **kwargs):

        if os.path.exists(self.soln_fn) and not rerun:
            self.soln = pickle.load(open(self.soln_fn, 'rb'))
            logging.info('DarkHistoryWrapper: Found existing DarkHistory initial conditions.')
            if 'dm_params' in self.soln and self.dm_params == self.soln['dm_params']:
                return self.soln
            else:
                logging.warning('DarkHistoryWrapper: DMParams mismatch, rerunning.')
        
        logging.info('DarkHistoryWrapper: Running DarkHistory to generate initial conditions...')
        default_kwargs = dict(
            DM_process=self.dm_params.mode, mDM=self.dm_params.m_DM,
            primary=self.dm_params.primary,
            sigmav=self.dm_params.sigmav, lifetime=self.dm_params.lifetime,
            struct_boost=self.dm_params.struct_boost,
            start_rs=3000, end_rs=end_rs, coarsen_factor=10, verbose=1,
        ) # default parameters use case B coefficients
        default_kwargs.update(kwargs)
        self.soln = evolve_DH(**default_kwargs)
        self.soln['dm_params'] = self.dm_params
        pickle.dump(self.soln, open(self.soln_fn, 'wb'))
        logging.info('DarkHistoryWrapper: Saved DarkHistory initial conditions.')
        return self.soln

    def get_init_cond(self, rs):
        """Returns initial conditions T_k [K], x_e [1], and photon bath spectrum [N / Bavg] at redshift rs=1+z."""

        T_k = np.interp(rs, self.soln['rs'][::-1], self.soln['Tm'][::-1] / phys.kB) # [K]
        x_e = np.interp(rs, self.soln['rs'][::-1], self.soln['x'][::-1, 0]) # HII

        spec_eng = self.soln['highengphot'][0].eng
        spec_N_arr = np.array([s.N for s in self.soln['highengphot']])
        spec_N = interpolate.interp1d(self.soln['rs'], spec_N_arr, kind='linear', axis=0, bounds_error=True)(rs) # [N / Bavg]
        spec = Spectrum(spec_eng, spec_N, rs=rs, spec_type='N')

        return T_k, x_e, spec


class TransferFunctionWrapper:
    """Wrapper for transfer functions from DarkHistory.

    Args:
        box_dim (int): Size of the box in pixels.
        abscs (dict): Abscissas.
        prefix (str, optional, TMP): Prefix for transfer function files.
        enable_elec (bool, optional): Enable electron injection. Default: True.
        on_device (bool, optional): Whether to save transfer function on device (GPU). Default: True.
    """
    
    def __init__(self, box_dim, abscs, prefix, enable_elec=True, on_device=True):
        
        self.box_dim = box_dim
        self.abscs = abscs
        self.prefix = prefix # temporary
        self.enable_elec = enable_elec
        self.on_device = on_device

        self.nBs_lowerbound = (1 + EPSILON) * np.min(self.abscs['nBs']) # [Bavg]
        self.load_tfs()
            
    def load_tfs(self):
        """Initialize transfer functions."""
        
        if not self.on_device:
            logging.warning('TransferFunctionWrapper: Not saving transfer functions on device!')
        self.phot_prop_tf = BatchInterpolator(f'{self.prefix}/phot_prop.h5', self.on_device)
        self.phot_scat_tf = BatchInterpolator(f'{self.prefix}/phot_scat.h5', self.on_device)
        self.phot_dep_tf  = BatchInterpolator(f'{self.prefix}/phot_dep.h5', self.on_device)
        logging.info('TransferFunctionWrapper: Loaded photon transfer functions.')
    
        if self.enable_elec:
            self.elec_scat_tf = BatchInterpolator(f'{self.prefix}/elec_scat.h5', self.on_device)
            self.elec_dep_tf  = BatchInterpolator(f'{self.prefix}/elec_dep.h5', self.on_device)
            logging.info('TransferFunctionWrapper: Loaded electron transfer functions.')
        else:
            logging.info('TransferFunctionWrapper: Skipping electron transfer functions.')
            
    def init_step(self, rs=..., delta_plus_one_box=..., x_e_box=...):
        """Initializes parameters and receivers for injection step."""

        delta_plus_one_box = np.clip(delta_plus_one_box, self.nBs_lowerbound, None)
        self.params = dict(
            rs = rs,
            nBs_box = delta_plus_one_box,
            x_e_box = x_e_box,
        )
        self.tf_kwargs = dict(
            rs = rs,
            nBs_s = delta_plus_one_box.ravel(),
            x_s = x_e_box.ravel(),
            out_of_bounds_action = 'clip',
        )
        self.prop_phot_N = np.zeros_like(self.abscs['photE']) # [N / Bavg]
        self.emit_phot_N = np.zeros_like(self.abscs['photE']) # [N / Bavg]
        self.dep_box = np.zeros((self.box_dim, self.box_dim, self.box_dim, len(self.abscs['dep_c']))) # [eV / Bavg]

    def inject_phot(self, in_spec, inject_type=..., weight_box=...):
        """Inject photons into (prop_phot_N,) emit_phot_N, and dep_box.

        Args:
            in_spec (Spectrum): Input photon spectrum.
            inject_type {'bath', 'ots', 'xray'}: Injection type.
            weight_box (ndarray): Injection weight box.
        """
        unif_norm = 1 / self.box_dim**3

        # Apply phot_prop_tf
        if inject_type == 'bath':
            sum_weight = None
            weight_norm = 1
            self.prop_phot_N += unif_norm * self.phot_prop_tf(
                in_spec=in_spec.N, sum_result=True, **self.tf_kwargs
            ) # [N / Bavg]
        elif inject_type == 'ots':
            sum_weight = weight_box.ravel()
            weight_norm = weight_box[..., None]
            self.emit_phot_N += unif_norm * self.phot_prop_tf(
                in_spec=in_spec.N, sum_result=True, sum_weight=sum_weight, **self.tf_kwargs
            ) # [N / Bavg]
        elif inject_type == 'xray':
            sum_weight = weight_box.ravel()
            weight_norm = weight_box[..., None]
        else:
            raise NotImplementedError(inject_type)
        
        # Apply phot_scat_tf
        self.emit_phot_N += unif_norm * self.phot_scat_tf(
            in_spec=in_spec.N, sum_result=True, sum_weight=sum_weight, **self.tf_kwargs
        ) # [N / Bavg]

        # Apply phot_dep_tf
        self.dep_box += weight_norm * self.phot_dep_tf(
            in_spec=in_spec.N, sum_result=False, **self.tf_kwargs
        ).reshape(self.dep_box.shape) # [eV / Bavg]

    def inject_elec(self, in_spec, weight_box=...):
        """Inject electrons into emit_phot_N and dep_box.

        Args:
            in_spec (Spectrum): Input electron spectrum.
            weight_box (ndarray): Injection weight box.
        """
        unif_norm = 1 / self.box_dim**3

        self.emit_phot_N += unif_norm * self.elec_scat_tf(
            in_spec=in_spec.N, sum_result=True, sum_weight=weight_box.ravel(), **self.tf_kwargs
        ) # [N / Bavg]
        self.dep_box += weight_box[..., None] * self.elec_dep_tf(
            in_spec=in_spec.N, sum_result=False, **self.tf_kwargs
        ).reshape(self.dep_box.shape) # [eV / Bavg]

    def inject_from_dm(self, dm_params, inj_per_Bavg_box):
        """Inject photons and electrons (on-the-spot) from dark matter.

        Args:
            dm_params (DMParams): Dark matter parameters.
            inj_per_Bavg_box (ndarray): Injection event per Bavg box.
        """
        if np.any(dm_params.inj_elec_spec.N != 0.) and not self.enable_elec:
            raise ValueError('Must enable electron injection.')
        
        self.inject_phot(dm_params.inj_phot_spec, inject_type='ots', weight_box=inj_per_Bavg_box)
        if self.enable_elec:
            self.inject_elec(dm_params.inj_elec_spec, weight_box=inj_per_Bavg_box)


    def populate_injection_boxes(self, input_heating, input_ionization, input_jalpha, dt):
        
        dep_heat_box = self.dep_box[...,3]
        dep_ion_box = (self.dep_box[...,0]/phys.rydberg + self.dep_box[...,1]/phys.He_ion_eng)
        dep_lya_box = self.dep_box[...,2]
        
        input_heating.input_heating += np.array(
            2 / (3*phys.kB*(1+self.params['x_e_box'])) * dep_heat_box / self.params['nBs_box'] / phys.A_per_B
        ) # [K/Bavg] / [B/Bavg] / [A/B] = [K/A]
    
        input_ionization.input_ionization += np.array(
            dep_ion_box / self.params['nBs_box'] / phys.A_per_B
        ) # [1/Bavg] / [B/Bavg] / [A/B] = [1/A]

        nBavg = phys.n_B * self.params['rs']**3 # [Bavg / pcm^3]
        dNlya_dVdt = dep_lya_box * nBavg / dt / phys.lya_eng # [lya pcm^-3 s^-1]
        nu_lya_Hz = (phys.lya_eng) / (2*np.pi*phys.hbar) # [Hz]
        J_lya = dNlya_dVdt * phys.c / (4*np.pi * nu_lya_Hz * phys.hubble(self.params['rs'])) # [lya pcm^-3 s^-1 pcm/s] / [sr Hz s^-1] = [lya sr^-1 s^-1 pcm^-2 Hz^-1]
        # hubble might be inconsistent with 1 / ((1+z) * dtdz)
        input_jalpha.input_jalpha += np.array(J_lya)

        self.params = None # invalidate parameters
        self.tf_kwargs = None # invalidate parameters

    @property
    def xray_eng_box(self):
        """X-ray energy-per-average-baryon box [eV / Bavg]."""
        return self.dep_box[..., 5]

    def attenuation_arr(self, rs, x, nBs=1.):
        dep_tf_at_point = self.phot_dep_tf.point_interp(rs=rs, x=x, nBs=nBs, out_of_bounds_action='clip')
        dep_toteng = np.sum(dep_tf_at_point[:, :5], axis=1) # H ion, He ion, exc, heat, cont
        return 1 - dep_toteng/self.abscs['photE']