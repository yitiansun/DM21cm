import os
import sys
import pickle
import numpy as np
from scipy import interpolate

WDIR = os.environ['DM21CM_DIR']
sys.path.append(WDIR)
import dm21cm.physics as phys
from dm21cm.utils import init_logger

sys.path.append(os.environ['DH_DIR'])
from darkhistory.main import evolve as evolve_DH
from darkhistory.spec.spectrum import Spectrum

logger = init_logger(__name__)
EPSILON = 1e-6


class DarkHistoryWrapper:
    """Wrapper for running DarkHistory prior to 21cmFAST steps.
    
    Args:
        injection (Injection): Injection object.
        prefix (str, optional): Prefix for DarkHistory initial conditions file.
        soln_name (str, optional): Name of DarkHistory initial conditions file.
    """
    
    def __init__(self, injection, prefix='.', soln_name='dh_init_soln.p'):
        self.injection = injection
        self.soln_fn = os.path.join(prefix, soln_name)

    def clear_soln(self):
        """Clears cached DarkHistory run."""
        if os.path.exists(self.soln_fn):
            logger.info('Removed cached DarkHistory run.')
            os.remove(self.soln_fn)

    def evolve(self, end_rs, rerun=False, **kwargs):
        """Runs DarkHistory to generate initial conditions.
        
        Args:
            end_rs (float): Final redshift rs = 1 + z.
            rerun (bool, optional): Whether to rerun DarkHistory. Default: False.
            **kwargs: Keyword arguments for DarkHistory evolve function.

        Returns:
            soln (dict): DarkHistory run solution.
        """
        if os.path.exists(self.soln_fn) and not rerun:
            self.soln = pickle.load(open(self.soln_fn, 'rb'))
            logger.info('Found existing DarkHistory initial conditions.')
            if 'injection' in self.soln and self.injection == self.soln['injection']:
                return self.soln
            else:
                logger.warning('Injection object mismatch, rerunning.')
        
        logger.info('Running DarkHistory to generate initial conditions...')

        # Custom injection API of DarkHistory
        start_rs = 3000
        coarsen_factor = 10

        def input_in_spec_phot(rs, next_rs=None, dt=None, state=None):
            """Injected photon spectrum per injection event [phot / inj]."""
            z_end = next_rs - 1 if next_rs is not None else None
            return self.injection.inj_phot_spec(rs-1, z_end=z_end, state=state) / self.injection.inj_rate(rs-1, z_end=z_end, state=state)
        
        def input_in_spec_elec(rs, next_rs=None, dt=None, state=None):
            """Injected electron spectrum per injection event [elec / inj]."""
            z_end = next_rs - 1 if next_rs is not None else None
            return self.injection.inj_elec_spec(rs-1, z_end=z_end, state=state) / self.injection.inj_rate(rs-1, z_end=z_end, state=state)
        
        def input_rate_func_N(rs, next_rs=None, dt=None, state=None):
            """Injection event rate density [inj / pcm^3 s]."""
            z_end = next_rs - 1 if next_rs is not None else None
            return self.injection.inj_rate(rs-1, z_end=z_end, state=state)
        
        def input_rate_func_eng(rs, next_rs=None, dt=None, state=None):
            """Injection power density [eV / pcm^3 s]."""
            z_end = next_rs - 1 if next_rs is not None else None
            return self.injection.inj_power(rs-1, z_end=z_end, state=state)

        default_kwargs = dict(
            in_spec_phot  = input_in_spec_phot, # [phot / inj]
            in_spec_elec  = input_in_spec_elec, # [elec / inj]
            rate_func_N   = input_rate_func_N,  # [inj / pcm^3 s]
            rate_func_eng = input_rate_func_eng, # [eV / pcm^3 s]
            start_rs = start_rs, end_rs = end_rs, coarsen_factor = coarsen_factor, verbose = 1,
            clean_up_tf = True,
        ) # default parameters use case B coefficients

        default_kwargs.update(kwargs)
        self.soln = evolve_DH(**default_kwargs)
        # self.soln['injection'] = self.injection
        pickle.dump(self.soln, open(self.soln_fn, 'wb'))
        logger.info('Saved DarkHistory initial conditions.')
        return self.soln

    def get_init_cond(self, rs):
        """Returns global averaged initial conditions for 21cmFAST.
        
        Args:
            rs (float): Matching redshift rs = 1 + z.

        Returns:
            T_k (float): Initial kinetic temperature [K].
            x_e (float): Initial ionization fraction [1].
            spec (Spectrum): Initial photon bath spectrum [N / Bavg].
        """

        T_k = np.interp(rs, self.soln['rs'][::-1], self.soln['Tm'][::-1] / phys.kB) # [K]
        x_e = np.interp(rs, self.soln['rs'][::-1], self.soln['x'][::-1, 0]) # HII

        spec_eng = self.soln['highengphot'][0].eng
        spec_N_arr = np.array([s.N for s in self.soln['highengphot']])
        spec_N = interpolate.interp1d(self.soln['rs'], spec_N_arr, kind='linear', axis=0, bounds_error=True)(rs) # [N / Bavg]
        spec = Spectrum(spec_eng, spec_N, rs=rs, spec_type='N')

        return T_k, x_e, spec
    
    # def get_evolve_z_s(self, start_rs, end_rs, coarsen_factor, dlnz=0.001):
    #     """Returns redshift array for DarkHistory injection steps, plus a final boundary z.
        
    #     Args:
    #         start_rs (float): Starting redshift rs = 1 + z.
    #         end_rs (float): Final redshift rs = 1 + z.
    #         coarsen_factor (int): Coarsening factor. See darkhistory.main.evolve.
    #         dlnz (float): Redshift log step size for redshift array. See darkhistory.main.evolve.

    #     Returns:
    #         z_s (array): Redshift array at which injection happens, plus a final boundary z.
    #     """
    #     rs_s = []
    #     rs = start_rs
    #     while rs > end_rs:
    #         rs_s.append(rs)
    #         rs = np.exp(np.log(rs) - dlnz * coarsen_factor)
    #     rs_s.append(rs)
    #     return np.array(rs_s) - 1