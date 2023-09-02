#!/bin/bash

#SBATCH --job-name=idl_elec_tf
#SBATCH --array=0-9
#SBATCH --partition=shared
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16GB
#SBATCH --time=0-08:00:00
#SBATCH --output=/n/home07/yitians/dm21cm/DM21cm/build_tf/slurm_outputs/%x_%j.out
#SBATCH --error=/n/home07/yitians/dm21cm/DM21cm/build_tf/slurm_outputs/%x_%j.err
#SBATCH --account=iaifi_lab
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yitians@mit.com

source /n/home07/yitians/setup_21cmfast.sh

cd /n/home07/yitians/dm21cm/DM21cm/build_tf

st=${SLURM_ARRAY_TASK_ID}
ed=$((${SLURM_ARRAY_TASK_ID} + 1))

srun idl <<< "gettf_nbs, i_xx_st=$st, i_xx_ed=$ed"