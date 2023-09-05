import os
import sys
import h5py
import numpy as np

from utils import load_dict


def save_aad(filename, axes, axes_abscs_keys, data):
    with h5py.File(filename, 'w') as hf:
        hf.create_dataset('axes', data=np.array(axes, dtype=h5py.string_dtype()))
        hf_abscs = hf.create_group('abscs')
        for axis, key in zip(axes, axes_abscs_keys):
            hf_abscs.create_dataset(axis, data=abscs[key])
        hf.create_dataset('data', data=data)


if __name__ == '__main__':

    #===== config =====
    run_name = '230629'
    make_list = ['elec_phot', 'elec_dep'] # {phot_phot, phot_dep, elec_phot, elec_dep}

    abscs = load_dict(f"../data/abscissas/abscs_{run_name}.h5")
    data_dir = os.environ['DM21CM_DATA_DIR'] + f'/tf/{run_name}'
    save_dir = os.environ['DM21CM_DATA_DIR'] + f'/tf/{run_name}'

    #===== phot -> phot =====
    if 'phot_phot' in make_list:

        print('phot_phot:')
        phot_tfgv_rxneo = np.load(f'{data_dir}/phot/phot_tfgv.npy')
        phot_phot = np.einsum('rxneo -> renxo', phot_tfgv_rxneo)
        save_aad(
            f"{save_dir}/phot/phot_phot.h5",
            ['rs', 'Ein', 'nBs', 'x', 'out'],
            ['rs', 'photE', 'nBs', 'x', 'photE'],
            phot_phot
        )

        print('phot_prop:')
        phot_prop = np.zeros_like(phot_tfgv_rxneo)
        for i_rs in range(len(abscs['rs'])):
            for i_x in range(len(abscs['x'])):
                for i_nBs in range(len(abscs['nBs'])):
                    np.fill_diagonal(phot_prop[i_rs,i_x,i_nBs], np.diagonal(phot_tfgv_rxneo[i_rs,i_x,i_nBs]))
        phot_prop = np.einsum('rxneo -> renxo', phot_prop)
        save_aad(
            f"{save_dir}/phot/phot_prop.h5",
            ['rs', 'Ein', 'nBs', 'x', 'out'],
            ['rs', 'photE', 'nBs', 'x', 'photE'],
            phot_prop
        )

        print('phot_prop_diag:')
        phot_prop_diag = np.zeros((len(abscs['rs']), len(abscs['x']), len(abscs['nBs']), len(abscs['photE'])))
        for i_rs in range(len(abscs['rs'])):
            for i_x in range(len(abscs['x'])):
                for i_nBs in range(len(abscs['nBs'])):
                    phot_prop_diag[i_rs,i_x,i_nBs] = np.diagonal(phot_tfgv_rxneo[i_rs,i_x,i_nBs])
        phot_prop_diag = np.einsum('rxno -> rnxo', phot_prop_diag) # o for both input and output
        save_aad(
            f"{save_dir}/phot/phot_prop_diag.h5",
            ['rs', 'nBs', 'x', 'out'],
            ['rs', 'nBs', 'x', 'photE'],
            phot_prop_diag
        )

        print('phot_scat:')
        phot_scat = phot_phot - phot_prop # renxo
        save_aad(
            f"{save_dir}/phot/phot_scat.h5",
            ['rs', 'Ein', 'nBs', 'x', 'out'],
            ['rs', 'photE', 'nBs', 'x', 'photE'],
            phot_scat
        )

        del phot_tfgv_rxneo, phot_phot, phot_prop, phot_prop_diag, phot_scat

    #===== elec -> phot =====
    if 'elec_phot' in make_list:
        print('elec_phot:')
        elec_tfgv_rxneo = np.load(f'{data_dir}/elec/elec_tfgv.npy')
        elec_scat = np.einsum('rxneo -> renxo', elec_tfgv_rxneo)
        save_aad(
            f"{save_dir}/elec/elec_scat.h5",
            ['rs', 'Ein', 'nBs', 'x', 'out'],
            ['rs', 'elecEk', 'nBs', 'x', 'photE'],
            elec_scat
        )

        del elec_tfgv_rxneo, elec_scat

    #===== phot -> dep =====
    if 'phot_dep' in make_list:
        print('phot_dep:')
        phot_depgv = np.load(data_dir + '/phot/phot_depgv.npy')
        phot_dep_Nf = np.einsum('rxneo -> renxo', phot_depgv)
        phot_dep_NE = np.einsum('renxo,e -> renxo', phot_dep_Nf, abscs['photE']) # multiply in energy
        save_aad(
            f"{save_dir}/phot/phot_dep.h5",
            ['rs', 'Ein', 'nBs', 'x', 'out'],
            ['rs', 'photE', 'nBs', 'x', 'dep_c'],
            phot_dep_NE
        )

        del phot_depgv, phot_dep_Nf, phot_dep_NE

    #===== elec -> dep =====
    if 'elec_dep' in make_list:
        print('elec_dep:')
        elec_depgv = np.load(data_dir + '/elec/elec_depgv.npy')
        elec_dep_Nf = np.einsum('rxneo -> renxo', elec_depgv)
        elec_dep_NE = np.einsum('renxo,e -> renxo', elec_dep_Nf, abscs['photE']) # multiply in energy
        save_aad(
            f"{save_dir}/elec/elec_dep.h5",
            ['rs', 'Ein', 'nBs', 'x', 'out'],
            ['rs', 'elecEk', 'nBs', 'x', 'dep_c'],
            elec_dep_NE
        )

        del elec_depgv, elec_dep_Nf, elec_dep_NE