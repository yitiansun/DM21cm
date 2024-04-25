#!/bin/bash

#SBATCH --job-name=decay-test
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

cd /n/home07/yitians/dm21cm/DM21cm/scripts

# export P21C_CACHE_DIR="/n/holylabs/LABS/iaifi_lab/Users/yitians/dm21cm/21cmFAST-cache" # if scratch is down
echo "CACHE: HOLYSCRATCH"
echo "DATA:  HOLYLABS"
echo "SAVE:  HOLYLABS & HOLYSCRATCH"

#python inj_script.py -r pwave-phot-test -c pwave-phot -i $SLURM_ARRAY_TASK_ID
python inj_script.py -r decay-test-2 -c decay-test-2 -i 0 -d 64

#SBATCH --constraint=cc8.0
#SABTCH --exclude=holygpu7c26305