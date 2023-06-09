"""Interpolators for energy depositions and secondary spectra."""

import pickle

import numpy as np
import jax.numpy as jnp
from jax.scipy.interpolate import RegularGridInterpolator
from jax import jit, vmap
from functools import partial


class BatchInterpolator:
    """Interpolator for multidimensional data. Currently support
    axes = ('rs', 'Ein', 'nBs', 'x', 'out')
    
    abscs : abscissas of axes.
    axes : tuple of name of axes.
    data : grid data consistent with axes and abscs.
    
    Initialize with tuple (abscs, axes, data) or a pickle filename containing
    this structure.
    """
    
    def __init__(self, abscs_axes_data):
        
        if isinstance(abscs_axes_data, str):
            abscs_axes_data = pickle.load(open(abscs_axes_data, 'rb'))
        self.abscs, self.axes, self.data = abscs_axes_data
        
        self.fixed_in_spec = None
        self.fixed_in_spec_data = None
        
        
    def set_fixed_in_spec(self, in_spec):
        
        self.fixed_in_spec = in_spec
        self.fixed_in_spec_data = jnp.einsum('e,renxo->rnxo', in_spec, self.data)
        
    
    def __call__(self, rs=None, in_spec=None, nBs_s=None, x_s=None,
                 sum_result=False, sum_weight=None, sum_batch_size=10000,
                 out_of_bounds_action='error'):
        """Batch interpolate in (nBs and) x directions.
        
        First sum with in_spec (with caching), then interpolate to a rs point,
        then perform the interpolation on [(nBs_s), x_s]. If sum_result is True,
        sum over all interpolated value.
        
        Parameters:
            rs : [1]
            in_spec : [N * ...]
            nBs_s : [1]
            x_s : [1]
            sum_result : if True, return average in the batch dimension.
            sum_weight : if None, just sum. otherwise dot.
            sum_batch_size : perform batch interpolation (and averaging) in batches of this size.
            out_of_bounds_action : {'error', 'clip'}
        
        Return:
            interpolated box or average
        """
        
        if out_of_bounds_action == 'clip':
            rs  = jnp.clip(rs,  jnp.min(self.abscs['rs']), jnp.max(self.abscs['rs']))
            x_s = jnp.clip(x_s, jnp.min(self.abscs['x']),  jnp.max(self.abscs['x']))
            nBs_s = jnp.clip(nBs_s, jnp.min(self.abscs['nBs']), jnp.max(self.abscs['nBs']))
        else:
            if not v_is_within(rs, self.abscs['rs']):
                raise ValueError('rs out of bounds.')
            if not v_is_within(x_s, self.abscs['x']):
                raise ValueError('x_s out of bounds.')
            if not v_is_within(nBs_s, self.abscs['nBs']):
                raise ValueError('nBs_s out of bounds.')
        
        ## 1. in_spec sum
        if jnp.all(in_spec == self.fixed_in_spec):
            in_spec_data = self.fixed_in_spec_data
        else:
            in_spec_data = jnp.einsum('e,renxo->rnxo', in_spec, self.data)
        
        ## 2. rs interpolation
        data_at_rs = interp1d(in_spec_data, self.abscs['rs'], rs)
        
        if not sum_result:
            
            ## 3. (nBs) x interpolation
            nBs_x_in = jnp.stack([nBs_s, x_s], axis=-1)
            return interp2d_vmap(
                data_at_rs,
                self.abscs['nBs'],
                self.abscs['x'],
                nBs_x_in
            )
            
        else:
            ## 3. (nBs) x sum
            split_n = int(jnp.ceil( len(x_s)/sum_batch_size ))
            if sum_weight is not None:
                sum_weight_batches = jnp.array_split(sum_weight, split_n)
                
            result = jnp.zeros( (len(self.abscs['out']),) ) # use numpy?
            
            nBs_x_in = jnp.stack([nBs_s, x_s], axis=-1)
            nBs_x_in_batches = jnp.array_split(nBs_x_in, split_n)

            for i_batch, nBs_x_in_batch in enumerate(nBs_x_in_batches):
                interp_result = interp2d_vmap(
                    data_at_rs,
                    self.abscs['nBs'],
                    self.abscs['x'],
                    nBs_x_in_batch
                )
                if sum_weight is None:
                    result += jnp.sum(interp_result, axis=0)
                else:
                    result += jnp.dot(sum_weight_batches[i_batch], interp_result)
                    
            return result