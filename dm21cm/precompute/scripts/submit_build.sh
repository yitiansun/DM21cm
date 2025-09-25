#!/bin/bash

#SBATCH --job-name=build-pbhacc-250909
#SBATCH --array=0-8
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32GB
#SBATCH --time=0-08:00:00
#SBATCH --output=/n/home07/yitians/dm21cm/DM21cm/outputs/slurm/%x_%a.out
#SBATCH --error=/n/home07/yitians/dm21cm/DM21cm/outputs/slurm/%x_%a.err
#SBATCH --account=iaifi_lab
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yitians@mit.com

source /n/home07/yitians/setup/dm21cm.sh

cd /n/home07/yitians/dm21cm/DM21cm/dm21cm/precompute/scripts


#===== hmf =====
# python build_hmf_tables.py

#===== pwave =====
# python build_pwave_tables.py

#===== pbhacc =====
# MODELS=("PRc23" "PRc14" "PRc29" "PRc23B" "PRc23H" "PRc23dm" "PRc23dp" "BHLl2")
MODELS=("PRc23H")
MVALS=(0.0 0.5 1.0 1.5 2.0 2.5 3.0 3.5 4.0)

# Compute the total number of combinations
NUM_MODELS=${#MODELS[@]}
NUM_MVALS=${#MVALS[@]}
TOTAL_JOBS=$((NUM_MODELS * NUM_MVALS))

# Map SLURM_ARRAY_TASK_ID to model and mval indices
MODEL_IDX=$((SLURM_ARRAY_TASK_ID / NUM_MVALS))
MVAL_IDX=$((SLURM_ARRAY_TASK_ID % NUM_MVALS))

echo "Running job for model: ${MODELS[$MODEL_IDX]}, log10mPBH: ${MVALS[$MVAL_IDX]}"
python build_pbhacc_tables.py --model ${MODELS[$MODEL_IDX]} --log10mPBH ${MVALS[$MVAL_IDX]}
