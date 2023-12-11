#!/bin/bash

#SBATCH --job-name=photdecay_hom
#SBATCH --array=1,8,10,12,13,14,15,16,17,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,37,42
#SBATCH --partition=gpu
#SABTCH --exclude=holygpu7c26305
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
python dm_script.py -i $SLURM_ARRAY_TASK_ID -z 002 -s 10 -c phot -r photdecay_hom --homogeneous

#--array=
# 0-23 for elec
# 0-43 for phot
# 0-3  for inhom

#SBATCH --constraint=cc8.0

# inhom fiducial values: photdecay mass 5e3, 1e25 s. elecdecay mass 1e7, 1e25 s.