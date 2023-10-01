import os
import sys
import h5py
import numpy as np

sys.path.append("..")
from dm21cm.utils import load_h5_dict


def save_aad(filename, axes, axes_abscs_keys, data):
    # use global abscs and dts
    with h5py.File(filename, 'w') as hf:
        hf.create_dataset('axes', data=np.array(axes, dtype=h5py.string_dtype()))
        hf_abscs = hf.create_group('abscs')
        for axis, key in zip(axes, axes_abscs_keys):
            hf_abscs.create_dataset(axis, data=abscs[key])
        hf.create_dataset('data', data=data)
        hf.create_dataset('dts', data=dts)


if __name__ == '__main__':

    #===== config =====
    run_name = 'zf01'
    make_list = ['phot_phot', 'phot_dep', 'elec_phot', 'elec_dep'] # {phot_phot, phot_dep, elec_phot, elec_dep}

    abscs = load_h5_dict(f"../data/abscissas/abscs_{run_name}.h5")
    data_dir = os.environ['DM21CM_DATA_DIR'] + f'/tf/{run_name}'
    save_dir = os.environ['DM21CM_DATA_DIR'] + f'/tf/{run_name}'

    dts = np.load(f'{data_dir}/phot/dt_rxneo.npy')[:, 1] # should be same for phot and elec

    #===== phot -> phot =====
    if 'phot_phot' in make_list:

        print('phot_phot', end=' ', flush=True)
        phot_tfgv_rxneo = np.load(f'{data_dir}/phot/phot_tfgv.npy')
        phot_phot = np.einsum('rxneo -> renxo', phot_tfgv_rxneo)
        save_aad(
            f"{save_dir}/phot_phot.h5",
            ['rs', 'Ein', 'nBs', 'x', 'out'],
            ['rs', 'photE', 'nBs', 'x', 'photE'],
            phot_phot
        )

        print('phot_prop', end=' ', flush=True)
        phot_prop = np.zeros_like(phot_tfgv_rxneo)
        for i_rs in range(len(abscs['rs'])):
            for i_x in range(len(abscs['x'])):
                for i_nBs in range(len(abscs['nBs'])):
                    np.fill_diagonal(phot_prop[i_rs,i_x,i_nBs], np.diagonal(phot_tfgv_rxneo[i_rs,i_x,i_nBs]))
        phot_prop = np.einsum('rxneo -> renxo', phot_prop)
        save_aad(
            f"{save_dir}/phot_prop.h5",
            ['rs', 'Ein', 'nBs', 'x', 'out'],
            ['rs', 'photE', 'nBs', 'x', 'photE'],
            phot_prop
        )

        print('phot_prop_diag', end=' ', flush=True)
        phot_prop_diag = np.zeros((len(abscs['rs']), len(abscs['x']), len(abscs['nBs']), len(abscs['photE'])))
        for i_rs in range(len(abscs['rs'])):
            for i_x in range(len(abscs['x'])):
                for i_nBs in range(len(abscs['nBs'])):
                    phot_prop_diag[i_rs,i_x,i_nBs] = np.diagonal(phot_tfgv_rxneo[i_rs,i_x,i_nBs])
        phot_prop_diag = np.einsum('rxno -> rnxo', phot_prop_diag) # o for both input and output
        save_aad(
            f"{save_dir}/phot_prop_diag.h5",
            ['rs', 'nBs', 'x', 'out'],
            ['rs', 'nBs', 'x', 'photE'],
            phot_prop_diag
        )

        print('phot_scat', end=' ', flush=True)
        phot_scat = phot_phot - phot_prop # renxo
        save_aad(
            f"{save_dir}/phot_scat.h5",
            ['rs', 'Ein', 'nBs', 'x', 'out'],
            ['rs', 'photE', 'nBs', 'x', 'photE'],
            phot_scat
        )

        del phot_tfgv_rxneo, phot_phot, phot_prop, phot_prop_diag, phot_scat

    #===== elec -> phot =====
    if 'elec_phot' in make_list:
        print('elec_phot', end=' ', flush=True)
        elec_tfgv_rxneo = np.load(f'{data_dir}/elec/elec_tfgv.npy')
        elec_scat = np.einsum('rxneo -> renxo', elec_tfgv_rxneo)
        save_aad(
            f"{save_dir}/elec_scat.h5",
            ['rs', 'Ein', 'nBs', 'x', 'out'],
            ['rs', 'elecEk', 'nBs', 'x', 'photE'],
            elec_scat
        )

        del elec_tfgv_rxneo, elec_scat

    #===== phot -> dep =====
    if 'phot_dep' in make_list:
        print('phot_dep', end=' ', flush=True)
        phot_depgv = np.load(data_dir + '/phot/phot_depgv.npy')
        phot_dep_Nf = np.einsum('rxneo -> renxo', phot_depgv)
        phot_dep_NE = np.einsum('renxo,e -> renxo', phot_dep_Nf, abscs['photE']) # multiply in energy
        save_aad(
            f"{save_dir}/phot_dep.h5",
            ['rs', 'Ein', 'nBs', 'x', 'out'],
            ['rs', 'photE', 'nBs', 'x', 'dep_c'],
            phot_dep_NE
        )

        del phot_depgv, phot_dep_Nf, phot_dep_NE

    #===== elec -> dep =====
    if 'elec_dep' in make_list:
        print('elec_dep', end=' ', flush=True)
        elec_depgv = np.load(data_dir + '/elec/elec_depgv.npy')
        elec_dep_Nf = np.einsum('rxneo -> renxo', elec_depgv)
        elec_dep_NE = np.einsum('renxo,e -> renxo', elec_dep_Nf, abscs['photE']) # multiply in energy
        save_aad(
            f"{save_dir}/elec_dep.h5",
            ['rs', 'Ein', 'nBs', 'x', 'out'],
            ['rs', 'elecEk', 'nBs', 'x', 'dep_c'],
            elec_dep_NE
        )

        del elec_depgv, elec_dep_Nf, elec_dep_NE

    print('done.')