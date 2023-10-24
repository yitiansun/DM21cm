# DM21cm

[Cross check run status](cross_check/xc_status.md)

## Assumptions
 - Using Planck18 dt everywhere (consistent with 21cmFAST)

## Debug deconstructor
 - [ ] Merge `debug` branch in 21cmFAST in `master`.
 - [ ] Remove `xraycheck`.
 - [ ] Restore commented out code near line 1960 in SpinTemperatureBox.c
 - [ ] Check bath is turned on.
 - [ ] In `dh_wrappers`, address all warnings.
 - [X] Make consistent `dt`.
 - [X] Change zf01-generated DarkHistory tf's nBs back to 1. (from 1.006)
 - [ ] Restore YHe values.
 - [ ] Slightly better treatment of bin 149 (10.16eV) photons.
 - [ ] Separate out powerbox.

## Transfer functions status
| name          | 0 | 1 | 2 | 3A | 3B | 3C | 3D | 4 phot | 4 elec | 3B DH |
|---------------|---|---|---|----|----|----|----|--------|--------|-------|
| zf01          | V | V | V | V  | V  | V  | V  |   V    |   V    |   V   |
| zf01-noHe     | / | / | B | B  | B  | B  | B  |   B    |   B    |   /   |
| zf001         | V | B | B | B  |    |    |    |   B    |        |   B   |
| zf001-noHe    | / | / | B | B  | B  | B  | B  |   B    |   B    |   /   |

- _: next
- R: running
- V: valid
- B: bugged: need to photoionization xsec


## Installation

### Prerequisite: DarkHistory
- Clone the DarkHistory repository [here](https://github.com/hongwanliu/DarkHistory/tree/test-dm21cm). Checkout branch `test-dm21cm`.
- Install the require packages via `pip install -r requirements.txt`.
- Set environment variable `DH_DIR` to point to the project folder (directory containing `README.md`).
- Set environment variable `DH_DATA_DIR` to point to the DarkHistory data (v1.1) folder (directory containing `binning.p`). The data files can be downloaded [here](https://zenodo.org/records/6819310).

### Prerequisite: 21cmFAST
- Clone the 21cmFAST fork [here](https://github.com/joshwfoster/21cmFAST). Checkout branch `master`.
- Install 21cmFAST according to README.md
  - Install gcc and have environment variable `CC` point to the binary.
  - Install gsl (GNU Scientific Library) and have `GSL_LIB` point to the directory of the library.
  - Install fftw and have `FFTW_INC` point to the directory containing fftw header files.
- Set environment variable `P21C_CACHE_DIR` to a directory for storing cached (requries at least 10G for a 128^3 box 1.01 redshift step run.)

### DM21cm
- Clone this repository.
- Download data files from [here]().
- Install the require packages via `pip install -r requirements.txt`.
- Set environment variable `DM21CM_DIR` to point to the project folder (directory containing `README.md`).
- Set environment variable `DM21CM_DATA_DIR` to point to the data folder (directory containing `phot_prop.h5`).