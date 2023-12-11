#!/bin/bash

#SBATCH --job-name=photdecay_halfstep
#SBATCH --array=0
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

source /n/home07/yitians/setup_dm21cm.sh

cd /n/home07/yitians/dm21cm/DM21cm/ProductionRuns

#python prod_run.py -i $SLURM_ARRAY_TASK_ID -c phot
python dm_script.py -i $SLURM_ARRAY_TASK_ID -z 002 -s 10 -c phot_halfstep -r photdecay_halfstep

#--array=
# 0-23 for elec
# 0-21 for phot
# 0-19 for phot_halfstep
# 0-3  for inhom

# inhom fiducial values: photdecay mass 5e3, 1e25 s. elecdecay mass 1e7, 1e25 s.