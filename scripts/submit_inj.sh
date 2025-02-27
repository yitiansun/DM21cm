#!/bin/bash

#SBATCH --job-name=pbh-acc-zm4
#SBATCH --array=0
#SBATCH --partition=iaifi_gpu
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

cd /n/home07/yitians/dm21cm/DM21cm/scripts

# export P21C_CACHE_DIR="/n/holylabs/LABS/iaifi_lab/Users/yitians/dm21cm/21cmFAST-cache" # if scratch is down
echo "CACHE: HOLYSCRATCH"
echo "DATA:  HOLYSTORE"
echo "SAVE:  HOLYSTORE & HOLYSCRATCH"

python inj_script.py -r pbh-acc-zm4 -c pbh-acc-PRc23 -i $SLURM_ARRAY_TASK_ID --multiplier_index 4
# python inj_script.py -r pbh-hr -c pbh-hr -i $SLURM_ARRAY_TASK_ID
#python inj_script.py -r xc-lt1e26-d128 -c decay-test -i 0

#SBATCH --constraint=cc8.0
#SBATCH --exclude=holygpu7c26305