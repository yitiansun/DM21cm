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

        if injection.mode not in ['DM decay']:
            raise NotImplementedError(injection.mode)

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
        default_kwargs = dict(
            DM_process='decay', mDM=self.injection.m_DM,
            primary=self.injection.primary,
            lifetime=self.injection.lifetime,
            start_rs=3000, end_rs=end_rs, coarsen_factor=10, verbose=1,
            clean_up_tf=True,
        ) # default parameters use case B coefficients
        default_kwargs.update(kwargs)
        self.soln = evolve_DH(**default_kwargs)
        self.soln['injection'] = self.injection
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