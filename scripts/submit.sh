#!/bin/bash

#SBATCH --job-name=pbhacc-MODEL-250909
#SBATCH --array=0-3
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

cd /n/home07/yitians/dm21cm/DM21cm/scripts

# export P21C_CACHE_DIR="/n/holylabs/LABS/iaifi_lab/Users/yitians/dm21cm/21cmFAST-cache" # if scratch is down
echo "CACHE: HOLYSCRATCH"
echo "DATA:  HOLYSTORE"
echo "SAVE:  HOLYSTORE & HOLYSCRATCH"

#--- bkg ---
# python bkg_script.py -r bkg -i $SLURM_ARRAY_TASK_ID
#--- pwave / pbhhr / pbhacc ---
# python inj_script.py --run_name pbhhr-a0.999-250909 --channel pbhhr-a0.999 -i $SLURM_ARRAY_TASK_ID --n_inj_steps 4 --step_mult 1
python inj_script.py --run_name pbhacc-MODEL-250909 --channel pbhacc-MODEL -i $SLURM_ARRAY_TASK_ID --n_inj_steps 4 --step_mult 1
#--- test ---
# python inj_script.py -r test0428 -c pbhacc-PRc23 -i 2 -d 32

#===== Unused sbatch options =====
#SBATCH --constraint=cc8.0
#SBATCH --exclude=holygpu7c26305