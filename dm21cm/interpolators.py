"""Interpolators for energy depositions and secondary spectra."""

import h5py

import numpy as np
from scipy import interpolate


class BatchInterpolator:
    """Interpolator for multidimensional data with axes ('rs', 'Ein', 'nBs', 'x', 'out')
    
    Args:
        filename (str): HDF5 data file name.

    Attributes:
        axes (list): List of axes.
        abscs (dict): Abscissas of axes.
        data (np.ndarray): Grid data consistent with axes and abscs.
        fixed_in_spec (np.ndarray): Fixed input spectrum.
        fixed_in_spec_data (np.ndarray): Data summed over fixed_in_spec.
    """

    def __init__(self, filename):

        with h5py.File(filename, 'r') as hf:
            self.axes = hf['axes'][:]
            self.abscs = {}
            for k, item in hf['abscs'].items():
                self.abscs[k] = item[:]
            self.data = hf['data'][:] # load into memory
        
        self.fixed_in_spec = None
        self.fixed_in_spec_data = None


    def set_fixed_in_spec(self, in_spec):
        
        self.fixed_in_spec = in_spec
        self.fixed_in_spec_data = np.einsum('e,renxo->rnxo', in_spec, self.data)


    def __call__(self, rs=..., in_spec=..., nBs_s=..., x_s=...,
                 sum_result=False, sum_weight=None, sum_batch_size=256**3,
                 out_of_bounds_action='error'):
        """Batch interpolate in nBs and x directions.
        
        First sum with in_spec (with caching), then interpolate to a rs point,
        then perform the interpolation on [nBs_s, x_s]. If sum_result is True,
        sum over all interpolated value.
        
        Args:
            rs (float): redshift [1]
            in_spec (1D array): input spectrum
            nBs_s (1D array): nBs values
            x_s (1D array): x values
            sum_result (bool): if True, sum over all interpolated value.
            sum_weight (optional, 1D array): weights for summing.
            sum_batch_size (int): perform interpolation in batches of this size.
            out_of_bounds_action {'error', 'clip'}
        
        Returns:
            interpolated box or sum of interpolated box.
        """
        
        if out_of_bounds_action == 'clip':
            rs  = np.clip(rs,  np.min(self.abscs['rs']), np.max(self.abscs['rs']))
            x_s = np.clip(x_s, np.min(self.abscs['x']),  np.max(self.abscs['x']))
            nBs_s = np.clip(nBs_s, np.min(self.abscs['nBs']), np.max(self.abscs['nBs']))
            bounds_error = False
        elif out_of_bounds_action == 'error':
            bounds_error = True
        else:
            raise NotImplementedError(out_of_bounds_action)
        
        # 1. in_spec sum
        if np.all(in_spec == self.fixed_in_spec):
            in_spec_data = self.fixed_in_spec_data
        else:
            in_spec_data = np.einsum('e,renxo->rnxo', in_spec, self.data)
        
        # 2. rs interpolation
        data_at_rs = interpolate.interp1d(self.abscs['rs'], in_spec_data, axis=0, copy=False)(rs) # (nBs, x, out)

        # 3. nBs x interpolation/sum
        interp = interpolate.RegularGridInterpolator(
            (self.abscs['nBs'], self.abscs['x']),
            data_at_rs,
            bounds_error = bounds_error,
            fill_value = np.nan,
        )
        nBs_x_in = np.stack([nBs_s, x_s], axis=-1)

        if not sum_result:
            return interp(nBs_x_in)
            
        else:
            split_n = int(np.ceil( len(x_s)/sum_batch_size ))
            if sum_weight is not None:
                sum_weight_batches = np.array_split(sum_weight, split_n)
                
            nBs_x_in_batches = np.array_split(nBs_x_in, split_n)
            result = np.zeros( (len(self.abscs['out']),) )

            for i_batch, nBs_x_in_batch in enumerate(nBs_x_in_batches):
                interp_result = interp(nBs_x_in_batch)
                if sum_weight is None:
                    result += np.sum(interp_result, axis=0)
                else:
                    result += np.dot(sum_weight_batches[i_batch], interp_result)
                    
            return result
        
    
    def point_interp(self, rs=..., nBs=..., x=...):
        """Returns the transfer function at a (rs, nBs, x) point."""

        interp = interpolate.RegularGridInterpolator(
            (self.abscs['rs'], self.abscs['nBs'], self.abscs['x']),
            np.einsum('renxo -> rnxeo', self.data),
            bounds_error = True,
        )
        return interp([rs, nBs, x])