import os
import sys

import numpy as np
from scipy import interpolate

sys.path.append(os.environ['DH_DIR'])
from darkhistory.config import load_data
import darkhistory.physics as phys


mass_threshold = {
    'elec_delta': phys.mass['e'], 'phot_delta': 0.,
    'e_L'   : phys.mass['e'],   'e_R': phys.mass['e'], 
    'e': phys.mass['e'],
    'mu_L'  : phys.mass['mu'], 'mu_R': phys.mass['mu'], 
    'mu': phys.mass['mu'],
    'tau_L' : phys.mass['tau'], 'tau_R' : phys.mass['tau'], 
    'tau': phys.mass['tau'],
    'q'     : 0.,             'c': phys.mass['c'],   
    'b'     : phys.mass['b'], 't': phys.mass['t'],
    'W_L': phys.mass['W'], 'W_T'   : phys.mass['W'], 'W': phys.mass['W'],
    'Z_L': phys.mass['Z'], 'Z_T'   : phys.mass['Z'], 'Z': phys.mass['Z'],
    'g': 0., 'gamma' : 0., 'h': phys.mass['h'],
    'nu_e': 0., 'nu_mu' : 0., 'nu_tau': 0.,
    'VV_to_4e'   : 2*phys.mass['e'], 'VV_to_4mu' : 2*phys.mass['mu'], 
    'VV_to_4tau' : 2*phys.mass['tau']
}

pri_to_ind = {
    'e_L': 0, 'e_R': 1, 'mu_L': 2, 'mu_R': 3, 'tau_L': 4, 'tau_R': 5,
    'q': 6, 'c': 7, 'b': 8, 't': 9,
    'W_L': 10, 'W_T': 11, 'Z_L': 12, 'Z_T': 13, 
    'g': 14, 'gamma': 15, 'h': 16,
    'nu_e': 17, 'nu_mu': 18, 'nu_tau': 19,
    'VV_to_4e': 20, 'VV_to_4mu': 21, 'VV_to_4tau': 22
}

dlNdlxIEW_interp = load_data('pppc')


def get_pppc_spec(eng_in=None, eng_out=None, pri=None, sec=None):
    """Get single PPPC spectrum.

    Args:
        eng_in (float): Total energy of the incoming particle in [eV].
        eng_out (array): Total energy spectrum of the outgoing particle in [eV].
        pri (str): Primary (input) channel.
        sec (str): Secondary (output) channel.

    Returns:
        array (array): dN/dE spectrum with units [parti/eV].
    """

    assert eng_in >= mass_threshold[pri]

    dN_dlog10x = 10**dlNdlxIEW_interp[sec][pri].get_val(eng_in/1e9, np.log10(eng_out/eng_in))

    return dN_dlog10x/(eng_out*np.log(10)) # dN/dE = dN/dlog10x * dlog10x/dE


def get_pppc_tf(eng_in=None, eng_out=None, pri=None, sec=None):
    """Get PPPC4DMID transfer function.

    Args:
        eng_in (array): Total energy spectrum of the incoming particle in [eV].
        eng_out (array): Total energy spectrum of the outgoing particle in [eV].
        pri (str): Primary (input) channel.
        sec (str): Secondary (output) channel.

    Returns:
        array (in, out): Transfer function (N -> dN/dE) with units [parti/eV / parti].
    """

    interp = dlNdlxIEW_interp[sec][pri] # interp takes (eng_in, log10x)
    eng_in_GeV = eng_in / 1e9
    log10x = np.log10(np.outer(1/eng_in, eng_out))

    # extrapolate x dependence to smaller (and larger) energies
    interp_E_min, interp_E_max = interp._mDM_in_GeV_arrs[0][0], interp._mDM_in_GeV_arrs[0][-1]
    eng_in_GeV_to_interp = np.clip(eng_in_GeV, interp_E_min, interp_E_max)

    dN_dlog10x_list = []
    for i in range(2):
        values_to_interp_x = interp._interpolators[i](eng_in_GeV_to_interp)
        log10_values = []
        for v, l in zip(values_to_interp_x, log10x):
            log10_value = np.full_like(l, -100.)
            ind_valid = (l >= interp._log10x_arrs[i][0]) & (l <= interp._log10x_arrs[i][-1])
            log10_value[ind_valid] = interpolate.PchipInterpolator(interp._log10x_arrs[i], v)(l[ind_valid])
            log10_values.append(log10_value)
        log10_values = np.array(log10_values)

        dN_dlog10x_list.append(interp._weight[i] * 10**log10_values)
    dN_dlog10x = np.sum(dN_dlog10x_list, axis=0)

    is_above_mass_threshold = eng_in >= mass_threshold[pri]

    return is_above_mass_threshold[:, None] * dN_dlog10x / (eng_out*np.log(10)) # dN/dE = dN/dlog10x * dlog10x/dE