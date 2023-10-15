# DM21cm

[Cross check run status](cross_check/xc_status.md)

## Assumptions
 - Using Planck18 dt everywhere (consistent with 21cmFAST)

## Debug deconstructor
 - [ ] Remove `xraycheck`
 - [ ] Restore commented out code near line 1960 in SpinTemperatureBox.c
 - [ ] Check bath is turned on.
 - [ ] In `dh_wrappers`, address all warnings.
 - [X] Make consistent `dt`
 - [X] Change zf01-generated DarkHistory tf's nBs back to 1. (from 1.006)
 - [ ] Restore YHe values
 - [ ] Slightly better treatment of bin 149 (10.16eV) photons.

## Transfer functions status
| name          | 0 | 1 | 2 | 3A | 3B | 3C | 3D | 4 phot | 4 elec | 3B DH |
|---------------|---|---|---|----|----|----|----|--------|--------|-------|
| zf01          | V | V | V | E  | E  | E  | E  |   E    |   E    |   V   |
| zf01-noHe     | / | / | V | V  |    |    |    |   V    |        |       |
| zf001         | V | V | V |    |    |    |    |        |        |   V   |
| zf001-noHe    | / | / | V | V  |    |    |    |   V    |        |       |

- _: next
- R: running
- V: valid
- E: need to fix excitation