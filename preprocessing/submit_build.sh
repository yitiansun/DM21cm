#!/bin/bash

#SBATCH --job-name=build-pbhacc-PRc50
#SBATCH --array=0-4
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

cd /n/home07/yitians/dm21cm/DM21cm/preprocessing

MVALS=(0.0 1.0 2.0 3.0 4.0)

python build_pbhacc_tables.py --model PRc50 --log10mPBH ${MVALS[$SLURM_ARRAY_TASK_ID]}