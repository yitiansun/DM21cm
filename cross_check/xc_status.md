## 3. Xray

[ ] xray_unif
- DC: `xc_xray_noLX_nos8_noHe_nosp`
- DH: `xc_xray_noHe`

[X] special
- DH: `xc_xrayST_noHe`
- DC: `xc_xray_noLX_nos8_noHe_nosp`
- DC: `xc_xray_noLX_nos8_noHe_nosp_noatten`
- DC: `xc_xray_noLX_noHe_nosp`
- DC: `xc_xray_noLX_noHe_nosp_noatten`
- 21: `xc_noHe_nosp`
- DC: `xc_xray_noLX_nos8_noHe_nosp_21totdep`

## 2. DM injection

[ ] phph_nos8_noHe_nosp_lifetime25
- DC: `xc_phph_noLX_nos8_noHe_nosp_lifetime25`
- DH: `xc_phph_noHe_lifetime25`

[R] phph_nos8_noHe_nosp_lifetime25_zf001
- DC: `xc_phph_noLX_nos8_noHe_nosp_lifetime25_zf001`
- DH: `xc_phph_noHe_lifetime25_zf001`

[ ] phph_lifetime25
- DC: `xc_phph_noLX_lifetime25`
- DH: `xc_phph_lifetime25`

[ ] ee_nos8_noHe_nosp_lifetime25
- DC: `xc_ee_noLX_nos8_noHe_nosp_lifetime25`
- DH: `xc_ee_noHe_lifetime25`

[ ] ee_nos8_noHe_nosp_lifetime25_zf001
- DC: `xc_ee_noLX_nos8_noHe_nosp_lifetime25_zf001`
- DH: `xc_ee_noHe_lifetime25_zf001`

[ ] ee_lifetime25
- DC: `xc_ee_noLX_lifetime25`
- DH: `xc_ee_lifetime25`

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