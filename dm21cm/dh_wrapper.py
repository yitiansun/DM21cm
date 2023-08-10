import numpy as np
import h5py, sys, os, pickle

sys.path.append("..")
from dm21cm.utils import load_dict
from dm21cm.data_loader import load_data


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