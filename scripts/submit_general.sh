#!/bin/bash

#SBATCH --job-name=zf001-elec-2
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

python 2_gather_fits.py --run_name zf001 --tf_type elec