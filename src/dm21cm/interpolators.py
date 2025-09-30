"""Interpolators for energy depositions and secondary spectra."""

import h5py
import numpy as np
from functools import partial

import jax.numpy as jnp
from jax import jit, vmap, device_put

EPSILON = 1e-6


#===== interpolation =====

@jit
def interp1d(fp, x0p, x0):
    """Interpolates f(x0), described by points fp and x0p, at values in x0.

    Args:
        fp (array): n(>=1)D array of function values. First dimension will be interpolated.
        x0p (array): 1D array of x0 values (first dimension of fp).
        x0 (array): x0 values to interpolate.

    Return:
        Interpolated values.

    Notes:
        x0p must be sorted. Does not do bound checks.
    """
    # --- locate lower indices along each axis -----------------
    i0l = jnp.searchsorted(x0p, x0, side='right') - 1

    # --- compute linear weights -------------------------------
    wl0 = (x0p[i0l + 1] - x0) / (x0p[i0l + 1] - x0p[i0l])
    wr0 = 1 - wl0

    # --- gather corner values ---------------------------------
    f0 = fp[i0l    ] * wl0
    f1 = fp[i0l + 1] * wr0

    return f0 + f1


@jit
def interp2d(fp, x0p, x1p, x01):
    """Interpolates f(x0, x1), described by points fp, x0p, and x1p, at values in x01.

    Args:
        fp (array): n(>=2)D array of function values. First two dimensions will be interpolated.
        x0p (array): 1D array of x0 values (first dimension of fp).
        x1p (array): 1D array of x1 values (second dimension of fp).
        x01 (array): [x0, x1] values to interpolate.

    Return:
        Interpolated values.

    Notes:
        x0p and x1p must be sorted. Does not do bound checks.
    """
    x0, x1 = x01

    # --- locate lower indices along each axis -----------------
    i0l = jnp.searchsorted(x0p, x0, side='right') - 1
    i1l = jnp.searchsorted(x1p, x1, side='right') - 1

    # --- compute linear weights -------------------------------
    wl0 = (x0p[i0l + 1] - x0) / (x0p[i0l + 1] - x0p[i0l])
    wl1 = (x1p[i1l + 1] - x1) / (x1p[i1l + 1] - x1p[i1l])
    wr0 = 1 - wl0
    wr1 = 1 - wl1

    # --- gather corner values ---------------------------------
    f00 = fp[i0l    , i1l    ] * wl0 * wl1
    f10 = fp[i0l + 1, i1l    ] * wr0 * wl1
    f01 = fp[i0l    , i1l + 1] * wl0 * wr1
    f11 = fp[i0l + 1, i1l + 1] * wr0 * wr1

    return f00 + f10 + f01 + f11

interp2d_vmap = vmap(interp2d, in_axes=(None, None, None, 0))


@jit
def interp3d(fp, x0p, x1p, x2p, x012):
    """Interpolates f(x0, x1, x2), described by points fp, x0p, x1p, and x2p at values in x012.

    Args:
        fp (array): n(>=2)D array of function values. First two dimensions will be interpolated.
        x0p (array): 1D array of x0 values (first dimension of fp).
        x1p (array): 1D array of x1 values (second dimension of fp).
        x2p (array): 1D array of x2 values (third dimension of fp).
        x012 (array): [x0, x1, x2] values to interpolate.

    Return:
        Interpolated values.

    Notes:
        x0p, x1p, and x2p must be sorted. Does not do bound checks.
    """
    x0, x1, x2 = x012

    # --- locate lower indices along each axis -----------------
    i0l = jnp.searchsorted(x0p, x0, side='right') - 1
    i1l = jnp.searchsorted(x1p, x1, side='right') - 1
    i2l = jnp.searchsorted(x2p, x2, side='right') - 1

    # --- compute linear weights -------------------------------
    wl0 = (x0p[i0l + 1] - x0) / (x0p[i0l + 1] - x0p[i0l])
    wl1 = (x1p[i1l + 1] - x1) / (x1p[i1l + 1] - x1p[i1l])
    wl2 = (x2p[i2l + 1] - x2) / (x2p[i2l + 1] - x2p[i2l])
    wr0 = 1 - wl0
    wr1 = 1 - wl1
    wr2 = 1 - wl2

    # --- gather corner values ---------------------------------
    f000 = fp[i0l    , i1l    , i2l    ] * wl0 * wl1 * wl2
    f100 = fp[i0l + 1, i1l    , i2l    ] * wr0 * wl1 * wl2
    f010 = fp[i0l    , i1l + 1, i2l    ] * wl0 * wr1 * wl2
    f110 = fp[i0l + 1, i1l + 1, i2l    ] * wr0 * wr1 * wl2
    f001 = fp[i0l    , i1l    , i2l + 1] * wl0 * wl1 * wr2
    f101 = fp[i0l + 1, i1l    , i2l + 1] * wr0 * wl1 * wr2
    f011 = fp[i0l    , i1l + 1, i2l + 1] * wl0 * wr1 * wr2
    f111 = fp[i0l + 1, i1l + 1, i2l + 1] * wr0 * wr1 * wr2

    return f000 + f100 + f010 + f110 + f001 + f101 + f011 + f111

interp3d_vmap = vmap(interp3d, in_axes=(None, None, None, None, 0))



#===== utilities =====

def bound_action(v, absc, out_of_bounds_action):
    """Check if v is within bounds of absc. If not, raise error or clip."""
    if out_of_bounds_action == 'clip':
        return jnp.clip(v, jnp.min(absc)*(1+EPSILON), jnp.max(absc)/(1+EPSILON))
    else:
        if not (jnp.all(v >= jnp.min(absc)) and jnp.all(v <= jnp.max(absc))):
            raise ValueError('value out of bounds.')
        return v


#===== interpolator class =====

class BatchInterpolator:
    """Interpolator for multidimensional data. Currently supports axes=('rs', 'Ein', 'nBs', 'x', 'out').

    Args:
        filename (str): HDF5 data file name.
        on_device (bool, optional): Whether to save data on device (GPU). Default: True.

    Attributes:
        axes (list): List of axis names.
        abscs (dict): Abscissas of axes.
        data (array): Grid data consistent with axes and abscs.
    """
    
    def __init__(self, filename, on_device=True):
        
        with h5py.File(filename, 'r') as hf:
            self.axes = hf['axes'][:]
            self.abscs = {}
            for k, item in hf['abscs'].items():
                if k == 'out':
                    self.abscs[k] = item[:]
                else:
                    self.abscs[k] = jnp.array(item[:])
            self.data = jnp.array(hf['data'][:]) # load into memory

        self.on_device = on_device
        if self.on_device:
            self.data = device_put(self.data)
            
        self.fixed_in_spec = None
        self.fixed_in_spec_data = None
    
    
    def set_fixed_in_spec(self, in_spec):
        """Set a fixed in_spec for faster interpolation.
        
        Args:
            in_spec (array): Input spectrum [N/Bavg].
        """
        self.fixed_in_spec = in_spec
        self.fixed_in_spec_data = jnp.einsum('e,renxo->rnxo', in_spec, self.data)
        if self.on_device:
            self.fixed_in_spec_data = device_put(self.fixed_in_spec_data)
        
        
    def __call__(self, rs=None, in_spec=None, nBs_s=None, x_s=None,
                 sum_result=False, sum_weight=None, sum_batch_size=128**3,
                 out_of_bounds_action='error'):
        """Batch interpolate in nBs and x.
        
        First sum with in_spec and interpolate to a rs point (with caching), then perform the
        interpolation on [nBs_s, x_s]. If sum_result is True, sum over all interpolated value
        with weight. The sum is done in batches for memory efficiency.
        
        Args:
            rs (float): Redshift rs=1+z values to interpolate to.
            in_spec (array): Input spectrum [N/Bavg].
            nBs_s (array): Relative baryon density values to interpolate to.
            x_s (array): Ionization fraction values to interpolate to.
            sum_result (bool): If True, sum over all interpolated value with weight.
            sum_weight (array): If not None, weight to sum over.
            sum_batch_size (int): Batch size for summing.
            out_of_bounds_action (str): 'error' or 'clip'.

        Return:
            Interpolated box or average of interpolated box.
        """

        rs = bound_action(rs, self.abscs['rs'], out_of_bounds_action)
        x_s = bound_action(x_s, self.abscs['x'], out_of_bounds_action)
        nBs_s = bound_action(nBs_s, self.abscs['nBs'], out_of_bounds_action)
        
        # 1. rs interp and in_spec sum
        if jnp.all(in_spec == self.fixed_in_spec):
            data_at_spec = self.fixed_in_spec_data # rnxo
            data_at_rs_at_spec = interp1d(data_at_spec, self.abscs['rs'], rs) # nxo
        else:
            data_at_rs = interp1d(self.data, self.abscs['rs'], rs) # enxo
            data_at_rs_at_spec = jnp.tensordot(in_spec, data_at_rs, axes=(0, 0)) # nxo
        
        # 2. (nBs) x interpolation (and sum)
        if not sum_result:
            
            nBs_x_in = jnp.stack([nBs_s, x_s], axis=-1)
            return interp2d_vmap(
                data_at_rs_at_spec,
                jnp.array(self.abscs['nBs']),
                jnp.array(self.abscs['x']),
                nBs_x_in
            )
        else:
            split_n = int(jnp.ceil( len(x_s)/sum_batch_size ))
            if sum_weight is not None:
                sum_weight_batches = jnp.array_split(sum_weight, split_n)
                
            result = jnp.zeros( (len(self.abscs['out']),) ) # use numpy?
            
            nBs_x_in = jnp.stack([nBs_s, x_s], axis=-1)
            nBs_x_in_batches = jnp.array_split(nBs_x_in, split_n)

            for i_batch, nBs_x_in_batch in enumerate(nBs_x_in_batches):
                interp_result = interp2d_vmap(
                    data_at_rs_at_spec,
                    self.abscs['nBs'],
                    self.abscs['x'],
                    nBs_x_in_batch
                )
                if sum_weight is None:
                    result += jnp.sum(interp_result, axis=0)
                else:
                    result += jnp.dot(sum_weight_batches[i_batch], interp_result)
                    
            return result
        
    
    def point_interp(self, rs=None, nBs=None, x=None, out_of_bounds_action='error'):
        """Returns the transfer function at a (rs, nBs, x) point.
        
        Args:
            rs (float): Redshift rs=1+z value to interpolate to.
            nBs (float): Relative baryon density value to interpolate to.
            x (float): Ionization fraction value to interpolate to.
            out_of_bounds_action (str): 'error' or 'clip'.

        Return:
            Interpolated transfer function.
        """

        rs = bound_action(rs, self.abscs['rs'], out_of_bounds_action)
        nBs = bound_action(nBs, self.abscs['nBs'], out_of_bounds_action)
        x = bound_action(x, self.abscs['x'], out_of_bounds_action)
        
        data = interp1d(self.data, self.abscs['rs'], rs) # enxo
        data = np.einsum('enxo -> nxeo', data) # nxeo
        data = interp1d(data, self.abscs['nBs'], nBs) # xeo
        data = interp1d(data, self.abscs['x'], x) # eo
        return data