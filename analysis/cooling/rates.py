import os
import sys
import numpy as np
from scipy import integrate

sys.path.append(os.environ['DH_DIR'])
import darkhistory.physics as dh_phys
import darkhistory.spec.spectools as spectools


def pp_CMB_rate(Ep, rs, **kwargs): # s^-1(eV, 1)
    
    # Ep: hard, ep: soft
    epmin = (1+1e-10) * dh_phys.me**2 / Ep
    epmax = 500*dh_phys.TCMB(rs)
    if epmax <= epmin:
        return 0
    eng_CMB = np.geomspace(epmin, epmax, 100) # eV
    eng_CMB_binbounds = spectools.get_bin_bound(eng_CMB) # eV
    eng_CMB_binwidths = eng_CMB_binbounds[1:] - eng_CMB_binbounds[:-1] # eV
    dndE_CMB = dh_phys.CMB_spec(eng_CMB, dh_phys.TCMB(rs)) # cm^-3 eV^-1
    dN_dEdt_arr = [] # eV^-1 s^-1
    
    for ep, dndE in zip(eng_CMB, dndE_CMB):
        
        E = Ep / dh_phys.me # 1(E/me) | input hard photon energy
        e = ep / dh_phys.me # 1(E/me) | CMB soft photon energy
        A = E + e
        
        def integrand(Ee): # 1(1) | Ee in units of me | numerical factor in parentheses in Eq.(C1) in 0906.1197
            return ( (4*A**2*np.log(4*e*Ee*(A-Ee)/A)) / (Ee*(A-Ee))
                     - 8*e*A + (2*(2*e*A-1)*A**2) / (Ee*(A-Ee))
                     - (1-1/(e*A))*(A**4)/(Ee**2*(A-Ee)**2) )
        # intg, err = integrate.quad(integrand, A/2*(1-np.sqrt(1-1/(E*e))), A/2*(1+np.sqrt(1-1/(E*e))))
        # if np.abs(err/intg) > 1e-5:
        #     raise ValueError('Large integration error.')
        xs = np.linspace(A/2*(1-np.sqrt(1-1/(E*e))), A/2*(1+np.sqrt(1-1/(E*e))), 100)
        intg = np.trapz(integrand(xs), xs)
        # dN/dEdt | eV^-1 s^-1 = cm^2 * cm/s * 1 * cm^-3 eV^-1
        dN_dEdt = dh_phys.thomson_xsec * dh_phys.c * (3/64) * (1/(e**2 * E**3)) * intg * dndE
        dN_dEdt_arr.append(dN_dEdt)
        
    rate = np.dot(dN_dEdt_arr, eng_CMB_binwidths) # s^-1
    return rate

def pp_matter_rate(Ep, rs, xHII=None, xHeII=None, xHeIII=None): # s^-1(eV, 1) | vectorized in Ep
    
    if xHII is None:
        xHII = dh_phys.xHII_std(rs)
    if xHeII is None:
        xHeII = dh_phys.xHeII_std(rs)
    if xHeIII is None:
        xHeIII = 0
        
    ref_rate = dh_phys.alpha * dh_phys.ele_rad**2 * dh_phys.c * dh_phys.nH * rs**3 # s^-1 = cm^2 * cm/s * cm^-3 | reference cross section
    
    # rate s^-1 = cm^2 * cm/s * cm^-3
    rate_xHII   = ref_rate * np.clip((28/9)*np.log(2*Ep/dh_phys.me)-(218/27), 0, None) * xHII
    rate_xHeII  = ref_rate * np.clip((28/9)*np.log(2*Ep/dh_phys.me)-(218/27), 0, None) * xHeII
    rate_xHeIII = ref_rate * np.clip((28/9)*np.log(2*Ep/dh_phys.me)-(218/27), 0, None) * xHeIII # placeholder
    rate_elec   = ref_rate * np.clip((28/9)*np.log(2*Ep/dh_phys.me)-(100/9),  0, None) * (xHII + xHeII + 2*xHeIII)
    rate_xHI    = ref_rate * 5.4  * np.log(513*Ep/(Ep+825*dh_phys.me)) * (1 - xHII)
    rate_xHeI   = ref_rate * 8.76 * np.log(513*Ep/(Ep+825*dh_phys.me)) * (dh_phys.nHe/dh_phys.nH - xHeII - xHeIII)
    return rate_xHII + rate_xHeII + rate_xHeIII + rate_elec + rate_xHI + rate_xHeI

def pp_matter_fullion_ZS_rate(Ep, rs, **kwargs): # s^-1(eV, 1)
    # s^-1 = cm^3 * cm^2 * cm s^-1
    rate = 20/3 * (dh_phys.nH+2*dh_phys.nHe) * rs**3 * dh_phys.alpha * dh_phys.ele_rad**2 * dh_phys.c * \
           (np.log(2*Ep/dh_phys.me) - 109/42)
    return np.clip(rate, 0, None)

def phph_scat_rate(Ep, rs, **kwargs): # s^-1(eV, 1) | vectorized in Ep
    # s^-1 = 1 * s^-1 * 1
    return 1.83e-27 * (2*dh_phys.h)**(-1) * (dh_phys.TCMB(1)/(2.7*dh_phys.kB))**6 \
           * dh_phys.H0 * rs**6 * (Ep/dh_phys.me)**3

def phph_scat_cool_rate(Ep, rs, **kwargs): # s^-1(eV, 1) | vectorized in Ep
    E = Ep/dh_phys.me
    def integrand(Ef): # 1(1)
        return (1-(Ef/E)+(Ef/E)**2)**2 * (E-Ef)
    intg, err = integrate.quad(integrand, E/2, E)
    if np.abs(err/intg) > 1e-5:
        raise ValueError('Large integration error.')
    return phph_scat_rate(Ep, rs) * (20/7) * intg / E**2

def compton_rate(Ep, rs, **kwargs): # s^-1(eV, 1)
    E = Ep/dh_phys.me
    def integrand(Ef): # 1(1)
        return Ef/E + E/Ef - 1 + (1 - (1/Ef - 1/E))**2
    intg, err = integrate.quad(integrand, E/(1+2*E), E)
    if np.abs(err/intg) > 1e-5:
        raise ValueError('Large integration error.')
    sigma = np.pi * dh_phys.alpha**2 * (dh_phys.ele_compton/(2*np.pi))**2 * (1/E**2) * intg # cm^2
    # if E < 1:
    #     sigma *= E
    return sigma * dh_phys.c * (dh_phys.nH + 2*dh_phys.nHe) * rs**3

def compton_cool_rate(Ep, rs, **kwargs): # s^-1(eV, 1)
    E = Ep/dh_phys.me
    def integrand(Ef): # 1(1)
        return (Ef/E + E/Ef - 1 + (1 - (1/Ef - 1/E))**2) * (E-Ef)
    intg, err = integrate.quad(integrand, E/(1+2*E), E)
    if np.abs(err/intg) > 1e-5:
        raise ValueError('Large integration error.')
    sigma = np.pi * dh_phys.alpha**2 * (dh_phys.ele_compton/(2*np.pi))**2 * (1/E**2) * intg / E # cm^2
    return sigma * dh_phys.c * (dh_phys.nH + 2*dh_phys.nHe) * rs**3

def photoion_rate(Ep, rs, xHII=None, xHeII=None): # s^-1(eV, 1)
    
    if xHII is None:
        xHII = dh_phys.xHII_std(rs)
    if xHeII is None:
        xHeII = dh_phys.xHeII_std(rs)
    
    def sigma_1e(E, Eth, Z): # cm^2(eV, eV)
        if E <= Eth:
            return 0
        eta = 1/(np.sqrt(E/Eth) - 1)
        return (2**9*np.pi**2*dh_phys.ele_rad**2)/(3*dh_phys.alpha**3*Z**2) * (Eth/E)**4 \
               * np.exp(-4*eta*np.arctan(1/eta)) / (1 - np.exp(-2*np.pi*eta))
    
    def sigma_HeII(E): # cm^2(eV)
        if E < 50: # eV
            return 0
        expn = -3.30 if E > 250 else -2.65
        return -12*sigma_1e(E, 13.6, 1) + 5.1e-20 * (E/250)**expn # cm^2
    
    rate_xHI   = sigma_1e(Ep, 13.6, 1) * dh_phys.c * dh_phys.nH * rs**3 * (1 - xHII)
    rate_xHeII = sigma_1e(Ep, 54.4, 2) * dh_phys.c * dh_phys.nH * rs**3 * xHeII
    rate_xHeI  = sigma_HeII(Ep) * dh_phys.c * dh_phys.nH * rs**3 * (dh_phys.nHe/dh_phys.nH - xHeII)
    
    return rate_xHI + rate_xHeII + rate_xHeI


# rates in Furlanetto 0910.4410
def ee_coll_rate(Eek, rs, xHII=None, use_tcool_approx=False): # [s^-1]([eV], [1], [1])
    
    xe = dh_phys.xHII_std(rs) + dh_phys.xHeII_std(rs) if xHII is None else xHII
    year = 365.25*86400 # [s]
    
    if use_tcool_approx:
        tcool = 5e3 * year * (1/xe) * (Eek/1e3)**(3/2) * (rs/10)**(-3)
        rate = 1 / tcool
    else:
        dEdt = dh_phys.elec_heating_engloss_rate(Eek, xe, rs) # [eV/s]
        rate = dEdt / Eek # [1/s]
        
    return rate

def e_ion_rate(Eek, rs, xHII=None): # [s^-1]([eV], [1], [1])
    year = 365.25*86400 # [s]
    xHI = 1 - (xHII if xHII is not None else dh_phys.xHII_std(rs))
    tcool = 5e5 * year * (1/xHI) * (Eek/1e3)**(3/2) * (rs/10)**(-3)
    return 1/tcool

def e_exc_rate(Eek, rs, xHII=None): # [s^-1]([eV], [1], [1])
    return e_ion_rate(Eek, rs, xHII=xHII)

# def e_ics_rate(Eek, rs, *args): # [s^-1]([eV], [1])
#     year = 365.25*86400 # [s]
#     gamma = (Eek + dh_phys.me)/dh_phys.me
#     tcool = 1e8 * year * (rs/10)**(-4) / gamma**2
#     return 1/tcool

def e_ics_rate(Eek, rs, *args): # [s^-1]([eV], [1])
    gamma = (Eek + dh_phys.me)/dh_phys.me
    beta = np.sqrt(1-1/gamma**2)
    rate = 4/3 * dh_phys.CMB_eng_density(dh_phys.TCMB(rs))*dh_phys.c*dh_phys.thomson_xsec * beta**2 * gamma**2/((gamma-1)*dh_phys.me)
    return rate