"""Some constants and functions used by all of dm21cm"""

import numpy as np
import os, sys

## units & constants
Mpc = 3.08568e24 # cm
c0 = 29979245800. # cm/s
fixed_cfdt = 0.6742 * 1 * Mpc / c0 # s

EPSILON = 1e-100


## abscissa

def photE_DH(n=500):
    dlnphoteng   = np.log(5565952217145.328/1e-4) / n
    photbins     = 1e-4 * np.exp(np.arange(n+1)*dlnphoteng)
    photenglow   = photbins[:n]
    photenghigh  = photbins[1:]
    photeng      = np.sqrt(photenglow * photenghigh)
    return photeng

def elecEk_DH(n=500):
    dlneng       = np.log(5565952217145.328)/n
    melec        = 510998.903
    elecbins     = melec + np.exp(np.arange(n+1) * dlneng)
    elecenglow   = elecbins[:n]
    elecenghigh  = elecbins[1:]
    eleceng      = np.sqrt((elecenglow - melec) * (elecenghigh - melec))
    return eleceng

abscs_nBs_test = {
    'nBs' : np.linspace(0., 2.7, 10),
    'x' : np.linspace(1e-5, 1-1e-5, 10),
    'rs' : np.logspace(np.log10(5.), np.log10(50.), 20),
    'photE' : photE_DH(n=500),
    'elecEk' : elecEk_DH(n=500),
    'dep_c' : ['H ion', 'He ion', 'exc', 'heat', 'cont']
}

abscs_nBs_test_2 = {
    'nBs' : np.linspace(0., 15., 10),
    'x' : np.linspace(1e-5, 1-1e-5, 10),
    'rs' : np.logspace(np.log10(5.), np.log10(100.), 20),
    'photE' : photE_DH(n=500),
    'elecEk' : elecEk_DH(n=500),
    'dep_c' : ['H ion', 'He ion', 'exc', 'heat', 'cont']
}


## utilities

def fitsfn(z, log10E, x, nBs, base=''):
    return f'{base}tf_z_{z:.3E}_logE_{log10E:.3E}_x_{x:.3E}_nBs_{nBs:.3E}.fits'