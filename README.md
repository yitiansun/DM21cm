# DM21cm - Inhomogeneous Energy Injection in 21cm Cosmology

[![arXiv](https://img.shields.io/badge/arXiv-2312.11608%20-green.svg)](https://arxiv.org/abs/2312.11608)

<p align="center"><img src="resources/logo.gif" /></p>

# Usage

```python
from dm21cm.injections.decay import DMDecayInjection
from dm21cm.evolve import evolve

import py21cmfast as p21c

return_dict = evolve(
    run_name = 'test_injection',
    z_start = 45.,
    z_end = 5.,
    injection = DMDecayInjection(
        primary = 'phot_delta',
        m_DM = 1e8, # [eV]
        lifetime = 1e28, # [s]
    ),
    p21c_initial_conditions = p21c.initial_conditions(
        user_params = p21c.UserParams(
            HII_DIM = 64,
            BOX_LEN = 256, # [conformal Mpc]
        ),
    ),
    p21c_astro_params = p21c.AstroParams(),
)
```

# Installation

### 1. Create a virtual environment
- We recommend creating a new environment for `DM21cm` and its dependencies. To do so via `conda`, run
```bash
conda create -n dm21cm python=3.12 pip
```

### 2. Install the modified 21cmFAST
- Clone the `21cmFAST` fork [here](https://github.com/joshwfoster/21cmFAST). Checkout branch `master`.
- Make sure `gcc` is available through the environment variable `CC`. This can be set via `export CC=/path/to/gcc/binary` in your `~/.bashrc` file, for example.
- Make sure the GNU Scientific Library (GSL) is avaiable. Set the environment variable `GSL_LIB` to the directory of the library.
- Make sure `fftw` is available. Set the environment variable `FFTW_INC` to the directory containing FFTW headers.
- Install `21cmFAST` from the project directory
```bash
pip install .
```
- Set the environment variable `P21C_CACHE_DIR` to a directory to store cache files.

### 3. Install DM21cm and DarkHistory (WIP)

- For GPU acceleration, install `jax>=0.4.14` according to your hardware specifications. See [JAX's repository](https://github.com/jax-ml/jax) for a guide. CPU-only installs can skip this step.
- Install `DM21cm` and associated packages (including `DarkHistory`) by
```bash
pip install dm21cm
```
- Download the data files required to run `DarkHistory` [here](), and set the environment variable `DH_DATA_DIR` to the directory.
- Download the data files required to run `DM21cm` [here](), and set the environment variable `DM21CM_DATA_DIR` to the directory.
- `DM21cm` should be available to run! You can test it with the example code above.


# Defining your custom injection

```python
import dm21cm.physics as phys
from dm21cm.injections.base import Injection
from darkhistory.spec import pppc
import numpy as np

class CustomInjection (Injection):

    def __init__(self):
        self.mode = 'Decay implemented again'
        self.primary = primary
        self.m_DM = m_DM
        self.lifetime = lifetime

    #===== injections =====
    def inj_rate(self, z):
        """Injection event rate density in [injection / pcm^3 s]. [pcm] = [physical cm].
        Used in DarkHistory part of the evolution.
        """
        rho_DM = phys.rho_DM * (1+z)**3 # [eV / pcm^3]
        return float((rho_DM/self.m_DM) / self.lifetime) # [inj / pcm^3 s]
    
    def inj_power(self, z):
        """Injection power density in [eV / pcm^3 s].
        Used in DarkHistory.
        """
        return self.inj_rate(z) * self.m_DM # [eV / pcm^3 s]
    
    def inj_phot_spec(self, z, **kwargs):
        """Injected photon rate density spectrum assuming a homogeneous universe in [# / pcm^3 s].
        Used in DarkHistory.
        """
        return self.phot_spec_per_inj * self.inj_rate(z) # [phot / pcm^3 s]
    
    def inj_elec_spec(self, z, **kwargs):
        """Injected electron rate density spectrum assuming a homogeneous universe in [# / pcm^3 s].
        Used in DarkHistory.
        """
        return self.elec_spec_per_inj * self.inj_rate(z) # [elec / pcm^3 s]
    
    def inj_phot_spec_box(self, z, delta_plus_one_box=..., **kwargs):
        """Injected photon rate density spectrum [# / pcm^3 s] and weight box [dimensionless]."""
        return self.inj_phot_spec(z), delta_plus_one_box # [phot / pcm^3 s], [1]

    def inj_elec_spec_box(self, z, delta_plus_one_box=..., **kwargs):
        """Injected electron rate density spectrum [# / pcm^3 s] and weight box [dimensionless]."""
        return self.inj_elec_spec(z), delta_plus_one_box # [elec / pcm^3 s], [1]

    #===== utilities =====
    def set_binning(self, abscs):
        """Inherent binning from `evolve` function."""
        self.phot_spec_per_inj = pppc.get_pppc_spec(
            self.m_DM, abscs['photE'], self.primary, 'phot', decay=True
        ) # [# / injection event]
        self.elec_spec_per_inj = pppc.get_pppc_spec(
            self.m_DM, abscs['elecEk'], self.primary, 'elec', decay=True
        ) # [# / injection event]

    def is_injecting_elec(self):
        """Optionally turn off electron injection."""
        return not np.allclose(self.elec_spec_per_inj.N, 0.)
    
    def get_config(self):
        """For caching darkhistory runs."""
        return {
            'mode': self.mode,
            'primary': self.primary,
            'm_DM': self.m_DM,
            'lifetime': self.lifetime
        }
```

# HERA sensitivity to dark matter monochromatic decays in 21-cm power spectrum
<img src="resources/limits.png" width="1000"/>

# Authors
Yitian Sun, Joshua W. Foster, Hongwan Liu, Julian B. Muñoz, and Tracy R. Slatyer
