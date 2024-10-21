import os
import sys

import numpy as np

WDIR = os.environ['DM21CM_DIR']
sys.path.append(WDIR)
from dm21cm.utils import load_h5_dict, save_h5_dict

from pppc import get_pppc_tf
from read import read_pbh


#===== main =====

def main():

    log10m_list = [15, 16.5, 18]

    for log10m in log10m_list:
        ddir = f"/n/home07/yitians/dm21cm/blackhawk/BlackHawk_v2.3/results/m{log10m:.3f}_pri"

        print(f'{log10m:.3f}', end=' ')
        hadronize(ddir)


#===== transfer function =====

dh_pri_dict = {
    'up' : 'q',
    'down' : 'q',
    'strange' : 'q',
    'charm' : 'c',
    'bottom' : 'b',
    'top' : 't',
    'wpm' : 'W',
    'z0' : 'Z',
    'higgs' : 'h',
    'gluon' : 'g',
    'muon' : 'mu',
    'tau' : 'tau',
}

def build_or_read_tf(tf_fn, eng_in, eng_out, force_rebuild=False):
    """Build or read the PPPC transfer functions.

    Args:
        tf_fn (str): The transfer function filename.
        eng_in (array): The incoming energy array [eV].
        eng_out (array): The outgoing energy array [eV].
    """

    if os.path.exists(tf_fn) and not force_rebuild:
        data = load_h5_dict(tf_fn)
        if np.allclose(data['eng_in'], eng_in) and np.allclose(data['eng_out'], eng_out):
            return data['tf']
        else:
            print('eng_in/eng_out mismatch!')

    print('building transfer functions...')
    tf = {'phot' : {}, 'elec' : {}}
    for pri, dh_pri in dh_pri_dict.items():
        print(pri, end='...')
        tf['phot'][pri] = get_pppc_tf(eng_in=eng_in, eng_out=eng_out, pri=dh_pri, sec='phot')
        tf['elec'][pri] = get_pppc_tf(eng_in=eng_in, eng_out=eng_out, pri=dh_pri, sec='elec')
        print('done.')

    data = {
        'tf' : tf,
        'eng_in' : eng_in,
        'eng_out' : eng_out,
    }
    save_h5_dict(tf_fn, data)
    print(f'Saved to {tf_fn}.')

    return tf


#===== utilities =====

def find_edge_log_uniform(arr):
    assert len(arr) >= 2
    arr_extended = np.concatenate(([ arr[0]**2/arr[1] ], arr, [ arr[-1]**2/arr[-2] ]))
    return np.sqrt(arr_extended[:-1] * arr_extended[1:])


def dE_from_E(eng):
    return np.diff(find_edge_log_uniform(eng))


def pack_data(dNdEdt, eng, t):
    data = np.zeros((dNdEdt.shape[0]+1, dNdEdt.shape[1]+1))
    data[1:, 1:] = dNdEdt * 1e9 # 1/eV s -> 1/GeV s
    data[0, 1:] = eng / 1e9 # eV -> GeV
    data[1:, 0] = t
    return data


#===== hadronize =====

def hadronize(ddir):
    """Hadronize the blackhawk output.

    Args:
        ddir (str): The directory containing the blackhawk output.
    """

    data_pri = read_pbh(ddir, 'primary', 'up') # for common E and t
    eng = data_pri['E'] # eV
    t = data_pri['t'] # s
    tf = build_or_read_tf('../../data/pppc/pppc_tf.h5', eng, eng)
    pri_list = [fn.split('_primary_')[0] for fn in os.listdir(ddir) if '_primary_' in fn]
    processed_list = []

    dNdEdt_phot = np.zeros((len(t), len(eng)))
    dNdEdt_elec = np.zeros((len(t), len(eng)))
    dNdEdt_nugr = np.zeros((len(t), len(eng)))

    for pri in pri_list:
        if pri in ['neutrinos', 'graviton']:
            data_pri = read_pbh(ddir, 'primary', pri)
            assert np.allclose(data_pri['E'], eng)
            dNdEdt_nugr += data_pri['dN_dEdt']
            processed_list.append(pri)
        elif pri == 'photon':
            data_pri = read_pbh(ddir, 'primary', 'photon')
            assert np.allclose(data_pri['E'], eng)
            dNdEdt_phot += data_pri['dN_dEdt']
            processed_list.append('photon')
        elif pri == 'electron':
            data_pri = read_pbh(ddir, 'primary', 'electron')
            assert np.allclose(data_pri['E'], eng)
            dNdEdt_elec += data_pri['dN_dEdt']
            processed_list.append('electron')
        else:
            data_pri = read_pbh(ddir, 'primary', pri)
            assert np.allclose(data_pri['E'], eng)
            dE = dE_from_E(eng)
            dNdEdt_phot += np.dot(data_pri['dN_dEdt'] * dE, tf['phot'][pri])
            dNdEdt_elec += np.dot(data_pri['dN_dEdt'] * dE, tf['elec'][pri])
            processed_list.append(pri)
    print('Processed:', processed_list)

    np.savetxt(f"{ddir}/photon_secondary_spectrum.txt", pack_data(dNdEdt_phot, eng, t), header='PPPC hadronization')
    np.savetxt(f"{ddir}/electron_secondary_spectrum.txt", pack_data(dNdEdt_elec, eng, t), header='PPPC hadronization')
    np.savetxt(f"{ddir}/nugr_secondary_spectrum.txt", pack_data(dNdEdt_nugr, eng, t), header='PPPC hadronization')

    return


if __name__ == '__main__':
    
    main()