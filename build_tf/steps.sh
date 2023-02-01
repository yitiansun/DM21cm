## step 1: generate fits file for each combination of (nBs, x, rs, injE)
# configure gettf_nbs.pro (adjust parallelizing)
# abscs are from gas_density_absc/gas_density_absc.ipynb section "abscs for IDL"
. part_nbs.sh | tqdms # parallelizes gettf_nbs.pro . ~15 processes work fine on erebus
# (tqdms is a multiprocess progress bar, or tqdm wrapper with --tqdm)

## step 2: combine fits outputs
# configure fits_to_array_nBs_A.py (adjust parallelizing)
. fits_to_array_nBs.sh | tqdms # paraleleizes fits_to_array_nBs_A.py
python fits_to_array_nBs_B.py

## step 3: process fits output to make transfer function grid values
python array_to_tfgv.py [--fixed_cfdt]

## step 4 (tmp): save grid_value with abscissa (multiple conventions)
# run rebuild_tfgv_to_ad.ipynb