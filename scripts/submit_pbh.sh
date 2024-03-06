#!/bin/bash

#SBATCH --job-name=p
#SBATCH --array=2,3,4,5,6,7
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

cd /n/home07/yitians/dm21cm/DM21cm/scripts

python pbh_script.py -r pbh -i $SLURM_ARRAY_TASK_ID

#--array=
# 0-7 for pbh
# 0-23 for elec
# 0-21 for elec_mid
# 0-43 for phot
# 0-3  for inhom

#SBATCH --constraint=cc8.0
#SABTCH --exclude=holygpu7c26305

# inhom fiducial values: photdecay mass 5e3, 1e25 s. elecdecay mass 1e7, 1e25 s.