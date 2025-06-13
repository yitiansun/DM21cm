import os
import numpy as np
import astropy.units as u
import astropy.constants as c

def output_specs(results_dir):
    fl = os.listdir(results_dir)
    return_dict = {'primary': [], 'secondary': []}

    for f in fl:
        if 'primary' in f:
            return_dict['primary'].append(f.split('_primary')[0])
        elif 'secondary' in f:
            return_dict['secondary'].append(f.split('_secondary')[0])
    
    return return_dict


def read_pbh(results_dir, data_type, particle=None):
    """Reads in the results from the blackhawk simulations.

    Args:
        results_dir (str): The directory where the results are stored.
        data_type {'evolution', 'primary', 'secondary'} : The type of data to read in.
        particle (str, optional): The particle type to read in.

    Returns:
        dict: A dictionary containing the results.
    """

    bh_data = np.genfromtxt(f"{results_dir}/BH_spectrum.txt", skip_header=1)
    rho_comov = bh_data[1, 1]
    assert np.isclose(rho_comov, 1.) # unitary density

    if data_type == 'evolution':
        t, M, a = np.loadtxt(f"{results_dir}/life_evolutions.txt", skiprows=4).T
        t, dt = np.loadtxt(f"{results_dir}/dts.txt", skiprows=2).T

        return {
            'M0' : bh_data[1, 0],
            'rho_comov' : bh_data[1, 1],
            't' : t,
            'dt' : dt,
            'M' : M,

            'units': {
                'M0' : u.g,
                'rho_comov' : u.cm**-3,
                't' : u.s,
                'dt' : u.s,
                'M' : u.g,
            }
        }
    
    elif data_type in ['primary', 'secondary']:
        data = np.genfromtxt(f"{results_dir}/{particle}_{data_type}_spectrum.txt", skip_header=1)

        return {
            'E' : 1e9 * data[0, 1:],
            't' : data[1:, 0],
            'dN_dEdt' : 1e-9 * data[1:, 1:], # row = t

            'units': {
                'E' : u.eV,
                't' : u.s,
                'dN_dEdt' : u.eV**-1 * u.s**-1,
            }
        }
    
    else:
        raise ValueError(f"Invalid data_type: {data_type}.")


def run_dir(version=None, log10m=None, hadron_code=None):
    return f"/n/home07/yitians/dm21cm/blackhawk/BlackHawk_{version}/results/m{log10m:.3f}_{hadron_code}"


def get_E_t(particle, pri_or_sec, **kwargs):
    rdir = run_dir(**kwargs)
    if particle == 'total':

        spec = read_pbh(rdir, 'primary', particle='graviton')
        spec_tot_eng = np.trapz(spec['dN_dEdt'] * spec['E'][None, :], x=spec['E'], axis=-1)

        if pri_or_sec == 'primary':
            p_list = [f.split('_')[0] for f in os.listdir(rdir) if '_primary_' in f and 'graviton' not in f]
        elif pri_or_sec == 'secondary':
            if kwargs['hadron_code'] in ['pythia']:
                p_list = ['photon', 'electron', 'proton', 'nu_e', 'nu_mu', 'nu_tau']
            elif kwargs['hadron_code'] in ['hazma']:
                p_list = ['photon', 'electron']
            elif kwargs['hadron_code'] in ['pri']:
                p_list = ['photon', 'electron', 'nugr']
            else:
                raise ValueError(f"Invalid hadron code.")
        else:
            raise ValueError(f"Invalid primary/secondary.")
        
        for p in p_list:
            spec = read_pbh(rdir, pri_or_sec, particle=p)
            spec_tot_eng += np.trapz(spec['dN_dEdt'] * spec['E'][None, :], x=spec['E'], axis=-1)
        spec_tot_eng = (spec_tot_eng * u.eV / c.c**2).to('g').value
    else:
        spec = read_pbh(rdir, pri_or_sec, particle=particle)
        spec_tot_eng = np.trapz(spec['dN_dEdt'] * spec['E'][None, :], x=spec['E'], axis=-1)
        spec_tot_eng = (spec_tot_eng * u.eV / c.c**2).to('g').value
    return spec['t'], spec_tot_eng