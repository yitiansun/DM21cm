# DM21cm status

# Run
BASE_DIR=/n/holystore01/LABS/iaifi_lab/Users/yitians/dm21cm/outputs/active/

## Current: up to version 250616
New vcb for pbhacc PS cosmo table
- pwave-phot-250630 [1.5-12]
    - RUNNING: [1.5, 5., 8.5, 12.]
    - DONE:
    - FISHED:
- pwave-elec-250630 [6.5-12]
    - RUNNING: [6.5, 8.5, 10.5, 12.]
    - DONE:
    - FISHED:
- pwave-tau-250630 [9.5-12]
    - RUNNING: [9.70, 10., 11., 12.]
    - DONE:
    - FISHED:
- pbhhr-a0 [13.5-18]
    - DATA: [13.5-18.0] at 0.25
    - RUNNING: 
    - DONE:
    - FISHED: [13.5-18.0] at 0.25
- pbhhr-a1 [13.5-18]
    - DATA: [13.5-18.0] at 0.25
    - RUNNING: 
    - DONE:
    - FISHED: [13.5-18.0] at 0.25

- pbhacc-PRc23/PRc14/PRc29/PRc23dp/PRc23dm/PRc23B/PRc23H/BHLl2-2506t30
    - DATA: 0 1 2 3 4
    - DONE: 0 2 4
    - FISHED: 0 2 4
BHLl2: in iter

# Data

## Version 250428
- hmf.h5
- pbhacc_rates/ (and cache)
    - PRc23_log10m[2.0]
    - PRc23R_log10m[2.0]
    - BHLl2_log10m[2.0]
- pwave_hmf_summed_rate.h5
- pbhhr.h5

# Version definitions
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

# Data generation
- `build_hmf_tables.py`: uses z, d, m, r.
    - hmf.h5: /n/holystore01/LABS/iaifi_lab/Users/yitians/dm21cm/data/hmf/
    - hmf_r.h5: Deprecated. With [250428]'s specification, too large. Not needed.
- `build_pbhacc_tables.py`: uses z, d, m, zfull_s, cinf_s
    - pbhacc_rates/: $WDIR/data/production/
- `build_pwave_tables.py`: uses z, d, m, zext_s
    - pwave_hmf_summed_rate.h5: $WIR/data/production/
- `build_pbhhr_tables.py`
    - pbhhr.h5:  $WIR/data/production/