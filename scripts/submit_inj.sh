#!/bin/bash

#SBATCH --job-name=pbhacc-BHLl2-iter0
#SBATCH --array=0-5
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

#--- pwave ---
# python inj_script.py -r pwave-elec-iter0 -c pwave-elec -i $SLURM_ARRAY_TASK_ID
#--- pbhacc ---
python inj_script.py -r pbhacc-BHLl2-iter0 -c pbhacc-BHLl2 -i $SLURM_ARRAY_TASK_ID
#--- pbhhr ---
# python inj_script.py -r pbhhr -c pbhhr -i $SLURM_ARRAY_TASK_ID
#--- test ---
# python inj_script.py -r test0428 -c pbhacc-PRc23 -i 2 -d 32

#===== Unused sbatch options =====
#SBATCH --constraint=cc8.0
#SBATCH --exclude=holygpu7c26305