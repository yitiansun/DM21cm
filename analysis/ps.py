import numpy as np
import powerbox as pbox

BOX_LEN = 256 # [Mpc]

def compute_power(
   box,
   length,
   n_psbins,
   log_bins=True,
   ignore_kperp_zero=True,
   ignore_kpar_zero=False,
   ignore_k_zero=False,
):
    # Determine the weighting function required from ignoring k's.
    k_weights = np.ones(box.shape, int)
    n0 = k_weights.shape[0]
    n1 = k_weights.shape[-1]

    if ignore_kperp_zero:
        k_weights[n0 // 2, n0 // 2, :] = 0
    if ignore_kpar_zero:
        k_weights[:, :, n1 // 2] = 0
    if ignore_k_zero:
        k_weights[n0 // 2, n0 // 2, n1 // 2] = 0

    res = pbox.tools.get_power(
        box,
        boxlength=length,
        bins=n_psbins,
        bin_ave=False,
        get_variance=False,
        log_bins=log_bins,
        k_weights=k_weights,
    )

    res = list(res)
    k = res[1]
    if log_bins:
        k = np.exp((np.log(k[1:]) + np.log(k[:-1])) / 2)
    else:
        k = (k[1:] + k[:-1]) / 2

    res[1] = k
    return res

def powerspectra(brightness_temp, z_start, z_end, n_psbins=30, logk=True):
    lightcone_redshifts = brightness_temp.lightcone_redshifts
    i_start = np.argmin(np.abs(lightcone_redshifts - z_start))
    i_end = np.argmin(np.abs(lightcone_redshifts - z_end))

    chunklen = (i_end - i_start) * brightness_temp.cell_size

    power, k = compute_power(
        brightness_temp.brightness_temp[:, :, i_start:i_end],
        (BOX_LEN, BOX_LEN, chunklen),
        n_psbins,
        log_bins=logk,
    )
    return {
        "k": k,
        "delta": power * k ** 3 / (2 * np.pi ** 2)
    }