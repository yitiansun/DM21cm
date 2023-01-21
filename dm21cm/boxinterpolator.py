import jax.numpy as jnp
from jax import jit, vmap
from functools import partial


##############################
## interpolating functions

def interp1d(f, x, xv):
    """Interpolates f(x) at values in xvs. Does not do bound checks.
    f : (n>=1 D) array of function value.
    x : 1D array of input value, corresponding to first dimension of f.
    xv : x values to interpolate.
    """
    li = jnp.searchsorted(x, xv) - 1
    lx = x[li]
    rx = x[li+1]
    p = (xv-lx) / (rx-lx)
    fl = f[li]
    return fl + (f[li+1]-fl) * p

#interp1d_vmap = vmap(interp1d, in_axes=(None, None, 0))


def interp2d(f, x0, x1, xv):
    """Interpolates f(x) at values in xvs. Does not do bound checks.
    f : (n>=2 D) array of function value.
    x0 : 1D array of input value, corresponding to first dimension of f.
    x1 : 1D array of input value, corresponding to second dimension of f.
    xv : [x0, x1] values to interpolate.
    """
    xv0, xv1 = xv
    
    li0 = jnp.searchsorted(x0, xv0) - 1
    lx0 = x0[li0]
    rx0 = x0[li0+1]
    p0 = (xv0-lx0) / (rx0-lx0)
    
    li1 = jnp.searchsorted(x1, xv1) - 1
    lx1 = x1[li1]
    rx1 = x1[li1+1]
    p1 = (xv1-lx1) / (rx1-lx1)
    
    fll = f[li0,li1]
    return fll + (f[li0+1,li1]-fll)*p0 + (f[li0,li1+1]-fll)*p1

interp2d_vmap = jit(vmap(interp2d, in_axes=(None, None, None, 0)))


##############################
## interpolator classes

# expect grid_data like (rs, Ein, nBs, x, c/Eout)

class DepositionBoxInterpolator:
    
    def __init__(self, grid_vals, abscs):
        self.grid_vals = grid_vals # (rs, in, nBs, x, c)
        self.abscs = abscs # should contain (rs, in, nBs, x, c)
        
        self.eng_dot = jit(lambda x, y: jnp.einsum('i,ijkl->jkl', x, y))
        
    #@partial(jit, static_argnums=(0,))
    def __call__(self, rs, eng_frac, nBs_s, x_s):
        
        ## 1. interpolate along rs
        grid_vals_at_rs = interp1d(self.grid_vals, self.abscs['rs'], rs)
        
        ## 2. apply spec
        grid_vals_to_interp = self.eng_dot(eng_frac, grid_vals_at_rs)
        #grid_vals_to_interp = jnp.einsum('i,ijkl->jkl', eng_frac, grid_vals_at_rs)
        #grid_vals_to_interp = jnp.sum(eng_frac * grid_vals_at_rs, axis=0)
        
        ## 3. interpolate along nBs and x
        nBs_x_in = jnp.stack([nBs_s, x_s], axis=-1)
        return interp2d_vmap(grid_vals_to_interp,
                             self.abscs['nBs'], self.abscs['x'], nBs_x_in)