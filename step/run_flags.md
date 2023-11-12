## Status

### 21cmFAST
[ ] `xunif`:      fcoll modification line ~1855: `fcoll = 1.;`
[ ] `xdecaytest`: fcoll modification line ~1855: `fcoll = (1.+curr_dens) / growth_factor_z * zpp_growth[R_ct];`
[X] `xdecay`:     fcoll modification line ~1855: `fcoll = (1.+curr_dens);`
[X] `nodplus1`: line ~1862, del_fcoll_Rct[box_ct] = fcoll;
[X] `x10`: injection multiplier 10
[X] `noxesink`: commented out `dxion_sink_dt`
[ ] `alldepion` : all deposition energy goes to ionization
[ ] `nosp`: species term OFF
[ ] `esf`: even-split-f
[ ] `noatten`: attenuation: OFF

flexible flags:
`nopop2`: turn off `Pop2_ion`
`zf001`: z-factor 1.001

### DarkHistory
[ ] `noHe`: YHe->0

### DM21cm
[ ] `noHe`: YHe->0
[X] sfrd: zf01: `1111`. zf001: `1111`.

base config: 32 64Mpc zf01 Rmax500

flexible flags:
`nodplus1`: similar. no 1+delta_R in fcoll
`xdecay`: custom cond_sfrd(z, delta, r) = 1+delta
`uddn`: uniform delta (in) deposition and normalization (normalization = energy to T and ionization level conversion)
`x[N]`: x N injection strength
`adashell[N]`: use N adaptive shells
`z[z]`: step init runs break at z

#### runs

xdecay_zf001

[ ] `stepinit_xdecay_ion_z10`
- Init: 21: xdecayx10 nodplus1 noxesink nopop2
- 21: xdecayx10 nodplus1 noxesink nopop2 alldepion
- DC: ...

[ ] `stepinit_dc_xdecay_ion_z10`
- Init: DC: xdecayx10 nodplus1 noxesink nopop2 uddn adashell40
- 21: xdecayx10 nodplus1 noxesink nopop2 alldepion
- DC: ...
