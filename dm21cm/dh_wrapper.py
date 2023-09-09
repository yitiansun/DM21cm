import numpy as np
import os, sys, pickle

# Import things we need from this work
sys.path.append("..")
import dm21cm.physics as phys
from dm21cm.data_loader import load_dict, load_data

# Import the PPPC Spectra
sys.path.append(os.environ['DH_DIR'])
from darkhistory.spec import pppc


class DMParams:
    """Dark matter parameters.
    
    Args:
        mode {'swave', 'decay'}: Type of injection.
        primary (str): Primary injection channel. See darkhistory.pppc.get_pppc_spec
        m_DM (float): DM mass in eV.
        abscs (dict): Abscissas.
        sigmav (float): Annihilation cross section in cm\ :sup:`3`\ s\ :sup:`-1`\ .
        lifetime (float, optional): Decay lifetime in s.
        
    Attributes:
        inj_phot_spec (Spectrum): Injected photon spectrum per injection event.
        inj_elec_spec (Spectrum): Injected electron positron spectrum per injection event.
        eng_per_inj (float): Injected energy per injection event.
    """
    
    def __init__(self, mode, primary, abscs, m_DM, sigmav=None, lifetime=None):
        
        if mode == 'swave':
            if sigmav is None:
                raise ValueError('must initialize sigmav.')
        elif mode == 'decay':
            if lifetime is None:
                raise ValueError('must initialize lifetime.')
        else:
            raise NotImplementedError(mode)
        
        self.mode = mode
        self.primary = primary
        self.m_DM = m_DM
        self.sigmav = sigmav
        self.lifetime = lifetime
        self.abscs = abscs
        
        self.inj_phot_spec = pppc.get_pppc_spec(
            self.m_DM, self.abscs['photE'], self.primary, 'phot',
            decay=(self.mode=='decay')
        )
        self.inj_elec_spec = pppc.get_pppc_spec(
            self.m_DM, self.abscs['elecEk'], self.primary, 'elec',
            decay=(self.mode=='decay')
        )
        self.eng_per_inj = self.m_DM if self.mode=='decay' else 2 * self.m_DM
        
    def __repr__(self):

        return f"DMParams(mode={self.mode}, primary={self.primary}, " \
            f"m_DM={self.m_DM:.4e}, sigmav={self.sigmav}, lifetime={self.lifetime})"


class DarkHistoryWrapper:
    """Wrapper for DarkHistory transfer functions.

    Args:
        box_dim (int): Size of the box in pixels.
        abscs (dict): Abscissas.
        tf_prefix (str, optional, TMP): Prefix for transfer function files.
        enable_elec (bool, optional): Enable electron injection. Default: True.
    """
    
    def __init__(self, box_dim, abscs, tf_prefix, enable_elec=True):
        
        self.box_dim = box_dim
        self.abscs = abscs
        self.tf_prefix = tf_prefix # temporary
        self.enable_elec = enable_elec

        self.load_tfs(tf_prefix)
            
    def load_tfs(self, tf_prefix, reload=False):
        """Initialize transfer functions."""
        
        self.phot_prop_tf = load_data('phot_prop', prefix=tf_prefix, reload=reload)
        self.phot_scat_tf = load_data('phot_scat', prefix=tf_prefix, reload=reload)
        self.phot_dep_tf  = load_data('phot_dep',  prefix=tf_prefix, reload=reload)
    
        if self.enable_elec:
            self.elec_scat_tf = load_data('elec_scat', prefix=tf_prefix, reload=reload)
            self.elec_dep_tf  = load_data('elec_dep',  prefix=tf_prefix, reload=reload)
            
    def init_step(self, rs=..., delta_plus_one_box=..., x_e_box=...):
        """Initializes parameters and receivers for injection step."""

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

    def get_state(self):
        """Returns prop_phot_N, emit_phot_N, and dep_box."""

        return self.prop_phot_N, self.emit_phot_N, self.dep_box

    def inject_phot(self, in_spec, inject_type=..., weight_box=...):
        """Inject photons into (prop_phot_N,) emit_phot_N, and dep_box.

        Args:
            in_spec (Spectrum): Input photon spectrum.
            inject_type {'bath', 'ots', 'xray'}: Injection type.
            weight_box (ndarray): Injection weight box.
        """
        norm = 1 / self.box_dim**3

        # Apply phot_prop_tf
        if inject_type == 'bath':
            sum_weight = None
            weight_norm = 1
            self.prop_phot_N += norm * self.phot_prop_tf(
                in_spec=in_spec.N, sum_result=True, **self.tf_kwargs
            ) # [N / Bavg]
        elif inject_type == 'ots':
            sum_weight = weight_box.ravel()
            weight_norm = weight_box[..., None]
            self.emit_phot_N += norm * self.phot_prop_tf(
                in_spec=in_spec.N, sum_result=True, sum_weight=sum_weight, **self.tf_kwargs
            ) # [N / Bavg]
        elif inject_type == 'xray':
            pass
        else:
            raise NotImplementedError(inject_type)
        
        # Apply phot_scat_tf
        self.emit_phot_N += norm * self.phot_scat_tf(
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
        norm = 1 / self.box_dim**3

        self.emit_phot_N += norm * self.elec_scat_tf(
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
        
        self.inject_phot_ots(dm_params.inj_phot_spec, injection_type='ots', weight_box=inj_per_Bavg_box)
        self.inject_elec_ots(dm_params.inj_elec_spec, injection_type='ots', weight_box=inj_per_Bavg_box)


    def populate_injection_boxes(self, input_heating, input_ionization, input_jalpha):
    
        # Populate input heating. [K/Bavg] / [B/Bavg] = [K/B]
        input_heating.input_heating += np.array(
            2 / (3*phys.kB*(1+self.params['x_e_box'])) * self.dep_box[...,3] / self.params['nBs_box']
        )
    
        # Populate input ionization. [1/Bavg] / [B/Bavg] = [1/B]
        input_ionization.input_ionization += np.array(
            (self.dep_box[...,0] + self.dep_box[...,1]) / phys.rydberg / self.params['nBs_box']
        )

        # Populate input lyman alpha
        nBavg = phys.n_B * self.params['rs']**3 # [Bavg / cm^3]
        n_lya = self.dep_box[...,2] * nBavg / phys.lya_eng # [lya cm^-3]
        dnu_lya = (phys.rydberg - phys.lya_eng) / (2*np.pi*phys.hbar) # [Hz^-1]
        J_lya = n_lya * phys.c / (4*np.pi) / dnu_lya # [lya cm^-2 s^-1 sr^-1 Hz^-1]

        input_jalpha.input_jalpha += np.array(J_lya)

        # Invalidate parameters
        self.params = None
        self.tf_kwargs = None