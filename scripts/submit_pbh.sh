#!/bin/bash

#SBATCH --job-name=pbh
#SBATCH --array=0-37
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

# export P21C_CACHE_DIR="/n/holylabs/LABS/iaifi_lab/Users/yitians/dm21cm/21cmFAST-cache"
echo "CACHE: HOLYSCRATCH"
echo "DATA:  HOLYLABS"
echo "SAVE:  HOLYLABS & HOLYSCRATCH"

python pbh_script.py -r pbh_fine -i $SLURM_ARRAY_TASK_ID

#SBATCH --constraint=cc8.0
#SABTCH --exclude=holygpu7c26305