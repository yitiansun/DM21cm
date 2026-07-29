import os
import numpy as np
from scipy import stats, interpolate


def get_limits(channel):
    """Get limits for a given channel.
    Meaning of inj: decay: 1/tau. pwave: c_sigma. pbh: f
    """
    data = np.loadtxt(os.environ['DM21CM_DIR'] + f"/outputs/limits/{channel}.txt", unpack=True)
    if len(data.shape) == 1:
        data = data[:, np.newaxis]
    mass_s, inj_s, sigma_s = data
    limit_s = np.sqrt(stats.chi2.ppf(.9, df=1)) * inj_s * sigma_s
    return mass_s, inj_s, sigma_s, limit_s

def get_limits_interp(channel):
    """Get limits interpolation function for a given channel."""
    mass_s, inj_s, sigma_s, limit_s = get_limits(channel)
    return interpolate.interp1d(mass_s, limit_s, kind="linear", bounds_error=True)