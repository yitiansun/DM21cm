# DM21cm - Inhomogeneous Energy Injection in 21cm Cosmology

[![arXiv](https://img.shields.io/badge/arXiv-2312.11608%20-green.svg)](https://arxiv.org/abs/2312.11608)
[![arXiv](https://img.shields.io/badge/arXiv-2509.XXXXX%20-green.svg)](https://arxiv.org/abs/2509.XXXXX)

<p align="center"><img src="resources/logo.gif" /></p>

# Usage

```python
from dm21cm.injections.dm import DMPWaveAnnihilationInjection
from dm21cm.evolve import evolve

import py21cmfast as p21c

return_dict = evolve(
    run_name = 'test',
    z_start = 45.,
    z_end = 5.,
    injection = DMPWaveAnnihilationInjection(
        primary = 'tau',
        m_DM = 1e10, # [eV]
        c_sigma = 1e-18, # [s]
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

### 3. Install DM21cm and DarkHistory

- For GPU acceleration, install `jax>=0.4.14` according to your hardware specifications. See [JAX's repository](https://github.com/jax-ml/jax) for a guide. CPU-only installs can skip this step.
- Install `DM21cm` and associated packages (including `DarkHistory`) by
```bash
pip install dm21cm
```
- Download the data files required to run `DarkHistory` [here](https://zenodo.org/records/13931543), and set the environment variable `DH_DATA_DIR` to the directory containing `binning.h5`.
- Download the data files required to run `DM21cm` [here](https://zenodo.org/records/10397814), and set the environment variable `DM21CM_DATA_DIR` to the directory containing `abscissas.h5`.
- `DM21cm` should be available to run! You can test it with the example code above, or notebooks in [examples](examples/).

### 4. Additional data tables for $p$-wave annihilating DM and PBH
- To run with $p$-wave annihilating DM or PBH Hawking radiation injection, download the additional data files [here](https://zenodo.org/records/17228967).
- To run PBH accretion injection, clone this repo and run [build_pbhacc_tables.py](src/dm21cm/precompute/scripts/build_pbhacc_tables.py) to build the required data tables. See also [this example](examples/3_custom_pbh_accretion.ipynb).


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

        # pre-compute spectrum for each injection event
        self.phot_spec_per_inj = pppc.get_pppc_spec(
            self.m_DM, abscs['photE'], self.primary, 'phot', decay=True
        ) # [# / injection event]
        self.elec_spec_per_inj = pppc.get_pppc_spec(
            self.m_DM, abscs['elecEk'], self.primary, 'elec', decay=True
        ) # [# / injection event]

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

# Authors
Yitian Sun, Joshua W. Foster, Hongwan Liu, Julian B. Muñoz, and Tracy R. Slatyer

# Citation

If you used `DM21cm` v1.1.0 in your work, please cite:
```bibtex
@article{sun2025constraining,
    title = "{Constraining inhomogeneous energy injection from annihilating dark matter and primordial black holes with 21-cm cosmology}",
    author = "Sun, Yitian and Foster, Joshua W. and Mu\~noz, Julian B.",
    eprint = "2509.XXXXX",
    archivePrefix = "arXiv",
    primaryClass = "hep-ph",
    month = "9",
    year = "2025"
}
```

If you used `DM21cm` v1.0.0 in your work, please cite:
```bibtex
@article{PhysRevD.111.043015,
    title = {Inhomogeneous energy injection in the 21-cm power spectrum: Sensitivity to dark matter decay},
    author = {Sun, Yitian and Foster, Joshua W. and Liu, Hongwan and Mu\~noz, Julian B. and Slatyer, Tracy R.},
    journal = {Phys. Rev. D},
    volume = {111},
    issue = {4},
    pages = {043015},
    numpages = {32},
    year = {2025},
    month = {Feb},
    publisher = {American Physical Society},
    doi = {10.1103/PhysRevD.111.043015},
    url = {https://link.aps.org/doi/10.1103/PhysRevD.111.043015}
}
```