import numpy as np

class Spectrum:
    """Simple spectrum structure inspired by DarkHistory.
    
    Main differences:
     - Reduced complexity (only supports N mode, i.e. particle number in bin).
     - Direct access to internals, N can be edited.
     - Restricted magic methods to only support adding Spectrum and multiplying scalars.
     - Redshift now uses z instead of 1+z to be consistent.
    
    Parameters
    ----------
    eng : ndarray
        Energies of bin centers.
    N : ndarray
        Photon number (per any unit, usually per averaged baryon number) in bin.
    z : float
        Redshift.
    """
    
    __array_priority__ = 1
    # __array_priority__ must be larger than 0, so that radd can work.
    # Otherwise, ndarray + Spectrum works by iterating over the elements of
    # ndarray first, which isn't what we want.

    def __init__(self, eng, N, z=None):

        if eng.shape != N.shape:
            raise ValueError('eng and N must have same shape.')
        self.eng = eng
        self.N = N
        self.z = z
        
    def toteng(self):
        return np.dot(self.eng, self.N)
    
    def redshift_to(self, z):
        raise NotImplementedError
        
    def copy(self):
        return Spectrum(self.eng, self.N, z=self.z)
        
    #===== magic methods =====
    def __repr__(self):
        return f'Spectrum(eng={self.eng}, N={self.N}, rs={self.rs})'
    
    def __add__(self, other):
        """Only supports Spectrum + Spectrum. Sum inherent self.z ."""
        if not np.all(self.eng == other.eng):
            raise ValueError('must have matching eng.')
        if self.z is not None and other.z is not None and self.z != other.z:
            raise ValueError('redshift mismatch.')
        return Spectrum(self.eng, self.N+other.N, z=self.z)
    
    def __radd__(self, other):
        return self + other
    
    def __neg__(self):
        return Spectrum(self.eng, -self.N, z=self.z)
    
    def __sub__(self, other):
        return self + (-other)
    
    def __rsub__(self, other):
        return other + (-self)
    
    def __mul__(self, other):
        """Only supports scalar multiplication."""
        if not np.isscalar(other):
            raise ValueError('can only multiply scalars with Spectrum.')
        return Spectrum(self.eng, self.N*other, z=self.z)
    
    def __rmul__(self, other):
        return self * other
    
    def __truediv__(self, other):
        """Only supports scalar division."""
        return self * (1/other)