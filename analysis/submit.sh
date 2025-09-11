#!/bin/bash

#SBATCH --job-name=fish
#SBATCH --array=0-16
#SBATCH --partition=shared
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

cd /n/home07/yitians/dm21cm/DM21cm/analysis

MODELS=(
    "pwave-phot-250909"
    "pwave-elec-250909"
    "pwave-tau-250909"
    "pwave-phot-mc1e11-250909"
    "pwave-elec-mc1e11-250909"
    "pwave-tau-mc1e11-250909"
    "pbhhr-a0.000-250909"
    "pbhhr-a0.999-250909"
    "pbhacc-PRc23-250909"
    "pbhacc-PRc14-250909"
    "pbhacc-PRc29-250909"
    "pbhacc-PRc23dm-250909"
    "pbhacc-PRc23dp-250909"
    "pbhacc-PRc23B-250909"
    "pbhacc-PRc23H-250909"
    "pbhacc-BHLl2-250909"
    "pbhacc-BHLl2mt-250909"
)

python fisher.py -r ${MODELS[$SLURM_ARRAY_TASK_ID]}