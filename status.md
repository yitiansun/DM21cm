# DM21cm status

# Data tables

## Version definitions

- pre250226:
    - `z_s = np.linspace(1, 50, 51)`
    - ...

- 250428: Extended and refined z_s in halo era
    - `z_s = np.geomspace(4, 100, 300)`
    - `d_s = np.linspace(-1, 1.6, 300)`
    - `m_s # [M_sun]` from `smi = SigmaMInterpSphere(res=4001)`, which is `np.geomspace(M_MIN, M_MAX, res)` with `M_MIN = 1` and `M_MAX = RHO_M * (4/3) * np.pi * 512**3`.
    - `r_s = np.geomspace(0.1, 10, 300) # [cMpc]`
    - `zfull_s = np.concatenate((z_s, np.geomspace(z_s[-1], 3000, 300)[1:]))`
    - `cinf_s = np.geomspace([~(1+0)*10K], [~(1+1)*1e4K], 128)`
    - `zext_s = np.concatenate((z_s, [z_s[-1]+1e-6, 4000]))`


## hmf (common) table : `build_hmf_tables.py`
Relevant ranges: z, d, m, r.

- hmf.h5
    - /n/holystore01/LABS/iaifi_lab/Users/yitians/dm21cm/data/hmf/: [250428]
- hmf_r.h5
    - Deprecated. With [250428]'s specification, too large. Not needed.

## pbhacc : `build_pbhacc_tables.py`
Relevant ranges: z, d, m, zfull_s, mPBH_s, cinf_s

- Cache: L_table_MODEL.h5
    - PRc23_log10m[2.000]
        - $WDIR/data/pbh-accretion/: [250428]

- pbhacc_rates/
    - PRc23_log10m[2.000]
        - $WDIR/data/production/: [250428]

## pwave : `build_pwave_tables.py`
Relevant ranges: z, d, m, zext_s

- pwave_hmf_summed_rate.h5
    - $WIR/data/production/: [250428]

## pbhhr : `build_pbhhr_tables.py`