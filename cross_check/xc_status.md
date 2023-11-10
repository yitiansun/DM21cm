## Status

21cmFAST:
[ ] `xunif`:      fcoll modification line ~1855: `fcoll = 1.;`
[ ] `xdecaytest`: fcoll modification line ~1855: `fcoll = (1.+curr_dens) / growth_factor_z * zpp_growth[R_ct];`
[X] `xdecay`:     fcoll modification line ~1855: `fcoll = (1.+curr_dens);`
[X] `nodplus1`: line ~1862, del_fcoll_Rct[box_ct] = fcoll;
[X] `x10`: injection multiplier 10
[X] `noxesink`: commented out `dxion_sink_dt`
[X] `alldepion` : all deposition energy goes to ionization
[ ] `nosp`: species term OFF
[ ] `esf`: even-split-f
[ ] `noatten`: attenuation: OFF


DM21cm:
sfrd: zf01: `old`. zf01: `old`.
physics: YHe: `ON`
custom_sfrd : 1+delta
`nodplus1`: ON

DarkHistory:
physics: YHe: `ON`

## 4. Xray spatial feature

base: 32_64Mpc
xunif: fcoll = cond_sfrd = 1.
nodplus1: del_fcoll_Rct[box_ct] = fcoll; instead of = (1+delta_R) * fcoll;
uddn: uniform delta (in) deposition and normalization (normalization = energy to T and ionization level conversion)
udd: uniform delta (in) deposition
x100: x100 injection strength
Rmax: max smoothing radius


[X] sf_xdecayx10_nodplus1_noxesink_nopop2_alldepion_uddn_01atten_varbath_ots
- DC: `sf_xdecayx10_nodplus1_dc_noLX_noxesink_nopop2_alldepion_uddn_01atten_bath32_ots`
- DC: `sf_xdecayx10_nodplus1_dc_noLX_noxesink_nopop2_alldepion_uddn_01atten_bath64_ots`
- DC: `sf_xdecayx10_nodplus1_dc_noLX_noxesink_nopop2_alldepion_uddn_01atten_bath128_ots`
- 21: `sf_xdecayx10_nodplus1_noxesink_nopop2_alldepion`

Working !!!!!!!
[X] sf_xdecayx10_nodplus1_noxesink_nopop2_alldepion_uddn_01atten_bath_ots
- DC: `sf_xdecayx10_nodplus1_dc_noLX_noxesink_nopop2_alldepion_uddn_01atten_bath_ots`
- 21: `sf_xdecayx10_nodplus1_noxesink_nopop2_alldepion`

[X] sf_xdecayx10_nodplus1_noxesink_nopop2_alldepion_uddn_01atten_bath_ots_zf001
- DC: `sf_xdecayx10_nodplus1_dc_noLX_noxesink_nopop2_alldepion_uddn_01atten_bath_ots_zf001`
- 21: `sf_xdecayx10_nodplus1_noxesink_nopop2_alldepion_zf001`


[X] sf_xdecayx10_nodplus1_noxesink_nopop2_uddn_01atten_bath_ots_zf001
- DC: `sf_xdecayx10_nodplus1_dc_noLX_noxesink_nopop2_uddn_01atten_bath_ots_zf001`
- 21: `sf_xdecayx10_nodplus1_noxesink_nopop2_zf001`


[X] sf_xdecayx100_nodplus1_nopop2_zf001_Rmax32
- DC: `sf_xdecayx100_nodplus1_dc_noLX_nopop2_uddn_zf001_Rmax32`
- 21: `sf_xdecayx100_nodplus1_nopop2_zf001_Rmax32`

[X] sf_xdecayx100_nodplus1_nopop2
- DC: `sf_xdecayx100_nodplus1_dc_noLX_nopop2_uddn_Rmax32`
- 21: `sf_xdecayx100_nodplus1_nopop2_Rmax32`

[X] sf_xdecayx100test_nodplus1_nopop2
- 21: `sf_xdecayx100test_nodplus1_nopop2`

[X] sf_xdecayx100_nodplus1_nopop2
- DC: `sf_xdecayx100_nodplus1_dc_noLX_nopop2_uddn`
- 21: `sf_xdecayx100_nodplus1_nopop2`

[X] sf_xunifx100_nodplus1_nopop2
- DC: `sf_xunifx100_nodplus1_dc_noLX_nopop2_uddn`
- 21: `sf_xunifx100_nodplus1_nopop2`

## 3. Xray

### 3.3 Convergence test

sfrd: new (PS. ST)

[X] ct_128_64Mpc_nopop2
- DC: `ct_128_64Mpc_xray_noLX_nopop2`
- 21: `ct_128_64Mpc_nopop2`

[X] ct_128_256Mpc_nopop2
- DC: `ct_128_256Mpc_xray_noLX_nopop2`
- 21: `ct_128_256Mpc_nopop2`

[X] ct_64_128Mpc_nopop2
- DC: `ct_64_128Mpc_xray_noLX_nopop2`
- 21: `ct_64_128Mpc_nopop2`

[X] ct_32_64Mpc_nopop2
- DC: `ct_32_64Mpc_xray_noLX_nopop2`
- DC: `ct_32_64Mpc_xray_noLX_nopop2_newPSSFRD`
- DC: `ct_32_64Mpc_xray_noLX_nopop2_newSFRD`
- 21: `ct_32_64Mpc_nopop2`

### 3.2 With 21cmFAST

[X] xc_xrayV21_nopop2_noHe_nosp_noatten_21totdep_esf
- DC: `xc_xray_noLX_nopop2_noHe_nosp_noatten_21totdep_esf`
- 21: `xc_nopop2_noHe_nosp_noatten_esf`

[X] xc_xrayV21_nopop2_nosp_noatten_esf
- DC: `xc_xray_noLX_nopop2_nosp_noatten_esf`
- 21: `xc_nopop2_nosp_noatten_esf`

[X] xc_xrayV21_nopop2_nosp_noatten
- DC: `xc_xray_noLX_nopop2_nosp_noatten`
- 21: `xc_nopop2_nosp_noatten`

[X] xc_xrayV21_nopop2
- DC: `xc_xray_noLX_nopop2`
- 21: `xc_nopop2`

[X] xc_xrayV21
- DC: `xc_xray_noLX`
- 21: `xc`

<!-- [B] xc_xrayV21_xHdep
- DC: `xc_xray_noLX_xHdep` using 1-xH as x for deposition instead of xe
- 21: `xc` -->

<!-- [B] xc_xrayV21_xeatten
- DC: `xc_xray_noLX_xeatten` using xe as x for attenuation instead of 1-xH
- 21: `xc` -->

### 3.1 With DarkHistory
[X] xc_xrayVDH_nos8_noHe_nosp
- DC: `xc_xray_noLX_nos8_noHe_nosp`
- DH: `xc_xrayST_noHe`

[X] xc_xrayVDH_nopop2_noHe_nosp
- DC: `xc_xray_noLX_nopop2_noHe_nosp`
- DH: `xc_xrayST_noHe`

[-] xc_xrayVDH_nopop2_noHe
- DC: `xc_xray_noLX_nopop2_noHe`
- DH: `xc_xrayST_noHe`

[X] xc_xrayVDH_nopop2
- DC: `xc_xray_noLX_nopop2`
- DH: `xc_xrayST`

[X] xc_xrayVDH
- DC: `xc_xray_noLX`
- DH: `xc_xrayST`

## 2. DM injection

### 2.1 Photon

[X] phph_nos8_noHe_nosp_lifetime25_zf001
- DC: `xc_phph_noLX_nos8_noHe_nosp_lifetime25_zf001`
- DH: `xc_phph_noHe_lifetime25_zf001`

[X] phph_nos8_noHe_nosp_lifetime25
- DC: `xc_phph_noLX_nos8_noHe_nosp_lifetime25`
- DH: `xc_phph_noHe_lifetime25`

[X] phph_nos8_noHe_lifetime25
- DC: `xc_phph_noLX_nos8_noHe_lifetime25`
- DH: `xc_phph_noHe_lifetime25`

[X] phph_nos8_lifetime25
- DC: `xc_phph_noLX_nos8_lifetime25`
- DH: `xc_phph_lifetime25`

[X] phph_nopop2_lifetime25
- DC: `xc_phph_noLX_nopop2_lifetime25`
- DH: `xc_phph_lifetime25`

[X] phph_lifetime25
- DC: `xc_phph_noLX_lifetime25`
- DH: `xc_phph_lifetime25`

### 2.2 Electron

[ ] ee_nos8_noHe_nosp_lifetime25
- DC: `xc_ee_noLX_nos8_noHe_nosp_lifetime25`
- DH: `xc_ee_noHe_lifetime25`

[ ] ee_nos8_noHe_nosp_lifetime25_zf001
- DC: `xc_ee_noLX_nos8_noHe_nosp_lifetime25_zf001`
- DH: `xc_ee_noHe_lifetime25_zf001`

[R] ee_nos8_lifetime26
- DC: `xc_ee_noLX_nos8_lifetime26`
- DH: `xc_ee_lifetime26`

[R] ee_lifetime26
- DC: `xc_ee_noLX_lifetime26`
- DH: `xc_ee_lifetime26`

## 1. Adiabatic evolution
DM21cm vs. DarkHistory

[X] base_nosp
- DC: `xc_noLX_nosp`
- DH: `xc_base`

[X] noHe_nos8_nosp
- DC : `xc_noLX_nos8_noHe_nosp`
- DH : `xc_noHe`

[X] noHe_nos8_nosp_zf001
- DC : `xc_noLX_nos8_noHe_nosp_zf001`
- DH : `xc_noHe_zf001`

[X] noHe_nosp
- DC : `xc_noLX_noHe_nosp`
- DH : `xc_noHe` (reuse)

[X] base
- DC : `xc_noLX`
- DH : `xc_base` (reuse)


## Baselines for cross checks

`21`: 21cmFAST.
`DC` : DM21cm.
`DH` : DarkHistory.

### 21cmFAST base
- z: 45 - 5
- z_factor (1+z)/(1+z_next): 1.01 (`tf_version`: zf01)
- no extra injection
- 32^3 box
- Planck 18 cosmology (including sigma 8) and `dt` from cosmo.age
- mass dependent zeta
- L_X = 40 (with default attenuation)
- YHe = 0.245
- pop2 stars ON
- species term present

### DarkHistory
- rs=1+z: 3000 - 4
- rs for new regime 0 transfer function: 49
- use case A alpha_recomb by turning on reionization with zero energy injection: `reion_switch=True`, `reion_rs=47`, `photion_rate` and `photoheat_rate` set to zero functions.
- `crosscheck_21cmfast=True` to use consistent tf, evolve order.

### DM21cm base
- (using above as defaults)
- L_X = 0, and using our own implementation