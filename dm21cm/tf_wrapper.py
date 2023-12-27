import os
import sys
import numpy as np
import jax.numpy as jnp

WDIR = os.environ['DM21CM_DIR']
sys.path.append(WDIR)
import dm21cm.physics as phys
from dm21cm.interpolators import BatchInterpolator
from dm21cm.utils import init_logger

logger = init_logger(__name__)
EPSILON = 1e-6


class TransferFunctionWrapper:
    """Wrapper for transfer functions.

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

        self.load_tfs()
        self.reset_phot()
        self.reset_dep()
            
    def load_tfs(self):
        """Load transfer functions from disk."""
        
        if not self.on_device:
            logger.warning('Not saving transfer functions on device!')
        self.phot_prop_tf = BatchInterpolator(f'{self.prefix}/phot_prop.h5', self.on_device)
        self.phot_scat_tf = BatchInterpolator(f'{self.prefix}/phot_scat.h5', self.on_device)
        self.phot_dep_tf  = BatchInterpolator(f'{self.prefix}/phot_dep.h5', self.on_device)
        logger.info('Loaded photon transfer functions.')
    
        if self.enable_elec:
            self.elec_scat_tf = BatchInterpolator(f'{self.prefix}/elec_scat.h5', self.on_device)
            self.elec_dep_tf  = BatchInterpolator(f'{self.prefix}/elec_dep.h5', self.on_device)
            logger.info('Loaded electron transfer functions.')
        else:
            logger.info('Skipping electron transfer functions.')
            
    def set_params(self, rs=..., delta_plus_one_box=..., x_e_box=..., T_k_box=..., homogenize_deposition=False):
        """Initializes parameters for deposition."""
        delta_plus_one_box = jnp.clip(
            delta_plus_one_box,
            (1 + EPSILON) * jnp.min(self.abscs['nBs']),
            (1 - EPSILON) * jnp.max(self.abscs['nBs'])
        )
        self.params = dict(
            rs = rs,
            nBs_box = delta_plus_one_box,
            x_e_box = x_e_box,
            T_k_box = T_k_box
        )
        if homogenize_deposition:
            self.params['nBs_box'] = jnp.ones_like(self.params['nBs_box']) * jnp.mean(self.params['nBs_box'])
            self.params['x_e_box'] = jnp.ones_like(self.params['x_e_box']) * jnp.mean(self.params['x_e_box'])
            self.params['T_k_box'] = jnp.ones_like(self.params['T_k_box']) * jnp.mean(self.params['T_k_box'])
        self.tf_kwargs = dict(
            rs = rs,
            nBs_s = self.params['nBs_box'].ravel(),
            x_s = self.params['x_e_box'].ravel(),
            out_of_bounds_action = 'clip',
        )

    def reset_phot(self):
        """Resets propagating and emission photon."""
        self.prop_phot_N = np.zeros_like(self.abscs['photE']) # [N / Bavg]
        self.emit_phot_N = np.zeros_like(self.abscs['photE']) # [N / Bavg]

    def reset_dep(self):
        """Resets deposition boxes and dt."""
        self.dep_box = np.zeros((self.box_dim, self.box_dim, self.box_dim, len(self.abscs['dep_c']))) # [eV / Bavg]
        self.dep_dt = 0.

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

    def increase_dt(self, dt):
        """Increase dep_dt by dt."""
        self.dep_dt += dt


    def populate_injection_boxes(self, input_heating, input_ionization, input_jalpha):
        """Populate input boxes for 21cmFAST and reset dep_box.
        
        Args:
            input_heating (InputHeating): Heating input box.
            input_ionization (InputIonization): Ionization input box.
            input_jalpha (InputJAlpha): Lyman-alpha input box.
            dt (float): Time in full cycle step [s].
        """
        
        dep_heat_box = self.dep_box[...,3]
        dep_ion_box = (self.dep_box[...,0]/phys.rydberg + self.dep_box[...,1]/phys.He_ion_eng)
        dep_lya_box = self.dep_box[...,2]
        
        delta_ionization_box = np.array(
            dep_ion_box / self.params['nBs_box'] / phys.A_per_B
        ) # [1/Bavg] / [B/Bavg] / [A/B] = [1/A]
        input_ionization.input_ionization += delta_ionization_box

        input_heating.input_heating += np.array(
            2 / (3*phys.kB*(1+self.params['x_e_box'])) * dep_heat_box / self.params['nBs_box'] / phys.A_per_B # [K/Bavg] / [B/Bavg] / [A/B] = [K/A]
            - self.params['T_k_box'] / (1+self.params['x_e_box']) * delta_ionization_box # species changing term [K] / [1] * [1/A] = [K/A]
        ) # here [K] just means [eV/kB], do not think of it as temperature

        nBavg = phys.n_B * self.params['rs']**3 # [Bavg / pcm^3]
        dNlya_dVdt = dep_lya_box * nBavg / self.dep_dt / phys.lya_eng # [lya pcm^-3 s^-1]
        nu_lya_Hz = (phys.lya_eng) / (2*np.pi*phys.hbar) # [Hz]
        J_lya = dNlya_dVdt * phys.c / (4*np.pi * nu_lya_Hz * phys.hubble(self.params['rs'])) # [lya pcm^-3 s^-1 pcm/s] / [sr Hz s^-1] = [lya sr^-1 s^-1 pcm^-2 Hz^-1]
        # hubble might be inconsistent with 1 / ((1+z) * dtdz)
        input_jalpha.input_jalpha += np.array(J_lya)

        assert not np.any(np.isnan(input_heating.input_heating)), 'input_heating has NaNs'
        assert not np.any(np.isnan(input_ionization.input_ionization)), 'input_ionization has NaNs'
        assert not np.any(np.isnan(input_jalpha.input_jalpha)), 'input_jalpha has NaNs'

        self.reset_dep()

    @property
    def xray_eng_box(self):
        """X-ray energy-per-average-baryon box [eV / Bavg]."""
        return self.dep_box[..., 5]
    
    @property
    def dep_box_means(self):
        """Deposition box means [eV / Bavg]."""
        return np.mean(self.dep_box, axis=(0,1,2))

    def attenuation_arr(self, rs, x, nBs=1.):
        """Attenuation (fraction of remaining) array w.r.t. energy.

        Args:
            rs (float): Redshift rs = 1 + z.
            x (float): Ionization fraction.
            nBs (float, optional): Relative baryon number density. Default: 1.

        Returns:
            atten (ndarray): Attenuation array.

        Notes:
            Applies to xray photons only (secondary photons not counted).
        """
        
        dep_tf_at_point = self.phot_dep_tf.point_interp(rs=rs, x=x, nBs=nBs, out_of_bounds_action='clip')
        dep_toteng = np.sum(dep_tf_at_point[:, :5], axis=1) # H ionization, He ionization, excitation, heating, into continuum, (into xray band excluded)
        return 1. - dep_toteng/self.abscs['photE']