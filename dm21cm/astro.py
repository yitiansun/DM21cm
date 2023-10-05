"Astrophysical xray injection class."

import numpy as np
from scipy import interpolate
from astropy import units as u


L_X_numerical_factor = 1e60 # make float happy


class AstroXray:
    

    def __init__(self, filename, abscs):

        self.data = np.load(filename, allow_pickle=True)
        self.z_range, self.delta_range, self.r_range = self.data['SFRD_Params']
        self.cond_sfrd_table = self.data['Cond_SFRD_Table']
        self.st_sfrd_table =  self.data['ST_SFRD_Table']

        # Takes the redshift as `z`
        # The overdensity parameter smoothed on scale `R`
        # The smoothing scale `R` in units of Mpc
        # Returns the conditional PS star formation rate density in [M_Sun / Mpc^3 / s]
        self.Cond_SFRD_Interpolator = interpolate.RegularGridInterpolator(
            (self.z_range, self.delta_range, self.r_range),
            self.cond_sfrd_table
        )

        # Takes the redshift as `z`
        # Returns the mean ST star formation rate density star formation rate density in [M_Sun / Mpc^3 / s]
        self.ST_SFRD_Interpolator = interpolate.interp1d(self.z_range, self.st_sfrd_table)

        self.abscs = abscs
        self.xray_eng_lo = 0.5 * 1000 # [eV]
        self.xray_eng_hi = 10.0 * 1000 # [eV]
        self.xray_i_lo = np.searchsorted(self.abscs['photE'], self.xray_eng_lo)
        self.xray_i_hi = np.searchsorted(self.abscs['photE'], self.xray_eng_hi)

    def get_spec(self, z):

        L_X_spec_prefac = 1e40 / np.log(4) * u.erg * u.s**-1 * u.M_sun**-1 * u.yr * u.keV**-1 # value in [erg yr / s Msun keV]
        L_X_spec_prefac /= L_X_numerical_factor
        # L_X (E * dN/dE) \propto E^-1
        L_X_dNdE = L_X_spec_prefac.to('1/Msun').value * (self.abscs['photE']/1000.)**-1 / self.abscs['photE'] # [1/Msun] * [1/eV] = [1/Msun eV]
        L_X_dNdE[:self.xray_i_lo] *= 0.
        L_X_dNdE[self.xray_i_hi:] *= 0.
        L_X_spec = Spectrum(self.abscs['photE'], L_X_dNdE, spec_type='dNdE', rs=1+z) # [1 / Msun eV]
        L_X_spec.switch_spec_type('N') # [1 / Msun]

        delta, L_X_spec
        emissivity_bracket = self.Cond_SFRD_Interpolator((z_donor, delta, R2))
        if np.mean(emissivity_bracket) > 0:
            emissivity_bracket *= (ST_SFRD_Interpolator(z_donor) / np.mean(emissivity_bracket))
        z_shell = z_edges[i_z_shell]
        emissivity_bracket *= (1 + delta) / (phys.n_B * u.cm**-3).to('Mpc**-3').value * dt
        emissivity_bracket *= L_X_numerical_factor * debug_xray_multiplier
        if xraycheck_is_box_average:
            i_xraycheck_loop_start = max(i_z_shell+1, i_xraycheck_loop_start)

        if 'xc-01attenuation' in debug_flags:
            L_X_spec_inj = L_X_spec.approx_attenuated_spectrum
            print_str += f'\n    approx attenuation: {L_X_spec.approx_attentuation_arr_repr[xray_i_lo:xray_i_hi]}'
        else:
            L_X_spec_inj = L_X_spec
        
        if ST_SFRD_Interpolator(z_donor) > 0.:
            tf_wrapper.inject_phot(L_X_spec_inj, inject_type='xray', weight_box=jnp.asarray(emissivity_bracket))