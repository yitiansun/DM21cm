#!/bin/bash

#SBATCH --job-name=elec_mid_hom
#SBATCH --array=2
#SBATCH --partition=iaifi_gpu_requeue
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

#python prod_run.py -i $SLURM_ARRAY_TASK_ID -c phot --stdastro
python dm_script.py -i $SLURM_ARRAY_TASK_ID -z 002 -s 10 -c elec -r elecdecay_mid_hom --homogeneous

#--array=
# 0-23 for elec
# 0-21 for elec_mid
# 0-43 for phot
# 0-3  for inhom

#SBATCH --constraint=cc8.0
#SABTCH --exclude=holygpu7c26305

# inhom fiducial values: photdecay mass 5e3, 1e25 s. elecdecay mass 1e7, 1e25 s.