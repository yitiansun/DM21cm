import numpy as np
import h5py, sys, os, pickle

# Import things we need from this work
sys.path.append("..")
import dm21cm.physics as phys
from dm21cm.data_loader import load_dict, load_data

# Import the PPPC Spectra
sys.path.append(os.environ['DH_DIR'])
from darkhistory.spec.pppc import get_pppc_spec

class DMParams:
    """Dark matter parameters.
    
    Args:
        mode {'swave', 'decay'}: Type of injection.
        primary (str): Primary injection channel. See darkhistory.pppc.get_pppc_spec
        m_DM (float): DM mass in eV.
        sigmav (float): Annihilation cross section in cm\ :sup:`3`\ s\ :sup:`-1`\ .
        lifetime (float, optional): Decay lifetime in s.
        
    Attributes:
        inj_phot_spec (Spectrum): Injected photon spectrum per injection event.
        inj_elec_spec (Spectrum): Injected electron positron spectrum per injection event.
        eng_per_inj (float): Injected energy per injection event.
    """
    
    def __init__(self, mode, primary, m_DM, abscs_path, sigmav=None, lifetime=None):
        
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
        abscs = load_dict(abscs_path)
        
        self.inj_phot_spec = get_pppc_spec(
            self.m_DM, abscs['photE'], self.primary, 'phot',
            decay=(self.mode=='decay')
        )
        self.inj_elec_spec = get_pppc_spec(
            self.m_DM, abscs['elecEk'], self.primary, 'elec',
            decay=(self.mode=='decay')
        )
        self.eng_per_inj = self.m_DM if self.mode=='decay' else 2 * self.m_DM
        
        
    def __repr__(self):
        
        return f"DMParams(mode={self.mode}, primary={self.primary}, "\
    f"m_DM={self.m_DM:.4e}, sigmav={self.sigmav}, lifetime={self.lifetime})"

class DarkHistoryWrapper:
    
    def __init__(self, HII_DIM, dhinit_list, z_step_factor,
                 dh_init_path, abscs_path, transfer_prefix,
                 enable_elec = False, force_reload_tf = False):
        
        # Basic run parameters
        self.HII_DIM = HII_DIM
        self.norm = 1/HII_DIM**3
        self.enable_elec = enable_elec
        self.force_reload_tf = force_reload_tf

        
        # Setting up the DH Transfer functions
        self.set_abscissa(z_step_factor, abscs_path)
        self.set_transfers(dhinit_list, transfer_prefix)
        
        # The DH initial condition that we will match
        self.dhinit_soln = pickle.load(open(dh_init_path, 'rb'))
    
    def set_abscissa(self, z_step_factor, abscs_path):
        
        '''
        This sets the abcissa for the photon energies and electron energies. We also
        set the size out the last dimension of the deposition box.
        '''
        
        abscs = load_dict(abscs_path)
        
        if not np.isclose(np.log(z_step_factor), abscs['dlnz']):
            raise ValueError('zplusone_step_factor and dhtf_version mismatch')
            
        self.photeng = abscs['photE']
        self.eleceng = abscs['elecEk']
        self.dep_size = len(abscs['dep_c'])
            
    def set_transfers(self, dhinit_list, transfer_prefix):
        '''
        This function initializes the DH transfer functions used in the 21cm run. 
        '''
                
        # Initialize photon transfer functions
        self.phot_prop_tf = load_data('phot_prop', prefix=transfer_prefix, reload=self.force_reload_tf)
        self.phot_scat_tf = load_data('phot_scat', prefix=transfer_prefix, reload=self.force_reload_tf)
        self.phot_dep_tf = load_data('phot_dep', prefix=transfer_prefix, reload=self.force_reload_tf)
        #self.phot_prop_tf.set_fixed_in_spec(dm_params.inj_phot_spec.N)
        #self.phot_scat_tf.set_fixed_in_spec(dm_params.inj_phot_spec.N)
        #self.phot_dep_tf.set_fixed_in_spec(dm_params.inj_phot_spec.N)
    
        # Initialize electron transfer functions if desired
        if self.enable_elec:
            raise NotImplementedError
            self.elec_phot_tf = load_data('elec_phot', prefix=transfer_prefix, reload=self.force_reload_tf)
            self.elec_dep_tf = load_data('elec_dep', prefix=transfer_prefix, reload=self.force_reload_tf)
            #self.elec_phot_tf.set_fixed_in_spec(dm_params.inj_elec_spec.N)
            #self.elec_dep_tf.set_fixed_in_spec(dm_params.inj_elec_spec.N)
            
    def set_empty_arrays(self):
        '''
        This is a convenience method that sets the properly shaped deposition box for 
        use in our evolution
        '''
            
        # Get the empty spectrum arrays
        self.prop_phot_N = np.zeros_like(self.photeng) # [N / Bavg]
        self.emit_phot_N = np.zeros_like(self.photeng) # [N / Bavg]

        # Get the empty deposition box. ('H ion', 'He ion', 'exc', 'heat', 'cont', 'xray')        
        self.dep_box = np.zeros((self.HII_DIM, self.HII_DIM, self.HII_DIM, self.dep_size))
        
        # We will also set the tf_kwargs to None to raise an error if they have not been reset
        self.tf_kwargs = None
        
    def set_tf_kwargs(self, tf_kwargs):
        '''
        Set the tf_kwargs
        '''
        
        self.tf_kwargs = tf_kwargs

    def get_state_arrays(self):

        '''
        This is a convenience method that provides the properly shaped deposition box for 
        use in our evolution
        '''
        
        # Return these
        return self.prop_phot_N, self.emit_phot_N, self.dep_box
    
    def photon_injection(self, spec_object, bath = True, weight_box = None, ots = False):
        if bath:
            self.prop_phot_N += self.norm*self.phot_prop_tf(in_spec=spec_object.N,
                                                            sum_result=True,
                                                            **self.tf_kwargs) # [N / Bavg]
            weight_box = np.ones((self.HII_DIM, self.HII_DIM, self.HII_DIM)) # dummy
            
        if ots:
            self.emit_phot_N += self.norm*self.phot_prop_tf(in_spec = spec_object.N, sum_result = True,
                                                            sum_weight = weight_box.ravel(), **self.tf_kwargs)
            
        self.emit_phot_N += self.norm*self.phot_scat_tf(in_spec=spec_object.N, sum_result=True,
                                                        sum_weight = weight_box.ravel(), **self.tf_kwargs)
        self.dep_box += weight_box[..., None] * self.phot_dep_tf(in_spec=spec_object.N, sum_result=False, **self.tf_kwargs).reshape(self.dep_box.shape) # [eV / Bavg]



    def populate_injection_boxes(self, input_heating, input_ionization, input_jalpha, x_e_box, delta_plus_one_box, nBavg):
    
        # Populate input heating. [K/Bavg] / [B/Bavg] = [K/B]
        input_heating.input_heating += np.array(2 / (3*phys.kB*(1+x_e_box)) * self.dep_box[...,3] / delta_plus_one_box)
    
        # Populate input ionization. [1/Bavg] / [B/Bavg] = [1/B]
        input_ionization.input_ionization += np.array((self.dep_box[...,0] + self.dep_box[...,1]) / phys.rydberg / delta_plus_one_box)

        # Populate input lyman alpha
        n_lya = self.dep_box[...,2] * nBavg / phys.lya_eng # [lya cm^-3]
        dnu_lya = (phys.rydberg - phys.lya_eng) / (2*np.pi*phys.hbar) # [Hz^-1]
        J_lya = n_lya * phys.c / (4*np.pi) / dnu_lya # [lya cm^-2 s^-1 sr^-1 Hz^-1]
        input_jalpha.input_jalpha += np.array(J_lya)
    
