#!/bin/bash

#SBATCH --job-name=evolve
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64GB
#SBATCH --time=0-08:00:00
#SBATCH --output=/n/home07/yitians/dm21cm/DM21cm/outputs/slurm/%x_%a.out
#SBATCH --error=/n/home07/yitians/dm21cm/DM21cm/outputs/slurm/%x_%a.err
#SBATCH --account=iaifi_lab
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yitians@mit.com

source /n/home07/yitians/setup_dm21cm.sh

cd /n/home07/yitians/dm21cm/DM21cm/scripts

python run_evolve.py > sf_xdecayx10_nodplus1_dc_noLX_noxesink_nopop2_alldepion_uddn_01atten_bath_ots_zf001.out