# DM21cm

![plot](notebooks/plots/DH.gif)

## Assumptions
 - Using Planck18 dt everywhere (consistent with 21cmFAST)

## Debug deconstructor
 - [ ] Remove `xraycheck`
 - [ ] Restore commented out code near line 1960 in SpinTemperatureBox.c
 - [ ] Check bath is turned on.
 - [ ] In `dh_wrappers`, address all warnings.
 - [ ] Make consistent `dt`
 - [ ] Change zf01-generated DarkHistory tf's nBs back to 1. (from 1.006)

## Transfer functions status
### valid
 - zf01
### need regeneration from step 2
 - zf05
 - zf001