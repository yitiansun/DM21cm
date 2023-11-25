# DM21cm
**A Semi-numerical Simulation of Inhomogeneous Energy Injection in 21cm**

![](plotting/logo/xH.gif)

## Installation

### Prerequisite: DarkHistory
- Clone the DarkHistory repository [here](https://github.com/hongwanliu/DarkHistory/tree/test-dm21cm). Checkout branch `DM21cm`.
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
- Set environment variable `DM21CM_DATA_DIR` to point to the data folder (directory containing `abscissas.h5`).