import numpy as np
from scipy import stats

limits_dir = "../outputs/limits"

# meaning of inj
# decay: 1/tau
# pwave: c_sigma
# pbh: f
def get_limits(m_s, channel):
    mass_s, inj_s, sigma_s = np.loadtxt(f"{limits_dir}/{channel}.txt", unpack=True)
    # for i_m, m in enumerate(mass_s):
    #     if m < mass_s[0] or m > mass_s[-1]:
    #         raise ValueError(f"i={i_m} m={m} out of range {mass_s[0]}-{mass_s[-1]}")
    
    limit_s = np.sqrt(stats.chi2.ppf(.9, df=1)) * inj_s * sigma_s
    
    return np.interp(m_s, mass_s, limit_s)