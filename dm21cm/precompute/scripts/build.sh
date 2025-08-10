#!/bin/bash

source /n/home07/yitians/setup/dm21cm.sh

cd /n/home07/yitians/dm21cm/DM21cm/dm21cm/precompute/scripts


#===== hmf =====
# python build_hmf_tables.py

#===== pwave =====
# python build_pwave_tables.py

#===== pbhacc =====
MODELS=("PRc23" "PRc14" "PRc29" "PRc23B" "PRc23H" "PRc23dm" "PRc23dp" "BHLl2")
MVALS=(0.0 1.0 2.0 3.0 4.0)

for ((i=0; i<${#MODELS[@]}; i++)); do
    for ((j=0; j<${#MVALS[@]}; j++)); do
        MODEL=${MODELS[$i]}
        MVAL=${MVALS[$j]}
        echo "Running job ${i}.${j}: model ${MODEL} (index ${i}), log10mPBH ${MVAL} (index ${j})"
        python build_pbhacc_tables.py --model ${MODEL} --log10mPBH ${MVAL}
    done
done