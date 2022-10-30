## step 1: generate fits file for each combination of (nBs, x, rs, injE)
# configure gettf_nbs.pro
. part_nbs.sh | tqdms # parallelizes gettf_nbs.pro . ~15 processes work fine on erebus
# (tqdms is a multiprocess tqdm wrapper)

## step 2: combine fits outputs
# configure fits_to_array_nBs_A.py
. fits_to_array_nBs.sh | tqdms # paraleleizes fits_to_array_nBs_A.py
python fits_to_array_nBs_B.py

## step 3: process fits output to make transfer function grid values
python array_to_tfgv.py