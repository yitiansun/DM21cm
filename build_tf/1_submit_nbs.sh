#!/bin/bash

#SBATCH --job-name=zf0002
#SBATCH --array=0-9
#SBATCH --partition=shared
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16GB
#SBATCH --time=0-08:00:00
#SBATCH --output=/n/home07/yitians/dm21cm/DM21cm/build_tf/slurm_outputs/%x_%a.out
#SBATCH --error=/n/home07/yitians/dm21cm/DM21cm/build_tf/slurm_outputs/%x_%a.err
#SBATCH --account=iaifi_lab
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yitians@mit.com

source /n/home07/yitians/setup_dm21cm.sh

cd /n/home07/yitians/dm21cm/DM21cm/build_tf

ISTART=${SLURM_ARRAY_TASK_ID}
IEND=$((${SLURM_ARRAY_TASK_ID} + 1))

# Avoid stupid IDL license lock
# FASRC Cannon: 1 second doesn't work, 10 seconds works fine, use 20 to be safe.
sleep $(($SLURM_ARRAY_TASK_ID * 20))

idl -e "gettf_nbs, i_xx_st=$ISTART, i_xx_ed=$IEND, run_name='zf0002', inj_mode='phot'" # full run
# idl -e "gettf_nbs, i_xx_st=$ISTART, i_xx_ed=$IEND, run_name='zfXXX', inj_mode='phot', i_nB_st=6, i_nB_ed=7" # only nBs=1