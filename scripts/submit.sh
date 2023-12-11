#!/bin/bash

#SBATCH --job-name=fc_xdecay_xesink_005
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=cc6.0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32GB
#SBATCH --time=0-08:00:00
#SBATCH --output=/n/home07/yitians/dm21cm/DM21cm/outputs/slurm/%x_%a.out
#SBATCH --error=/n/home07/yitians/dm21cm/DM21cm/outputs/slurm/%x_%a.err
#SBATCH --account=iaifi_lab
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yitians@mit.com

source /n/home07/yitians/setup_test-dm21cm.sh

cd /n/home07/yitians/dm21cm/DM21cm/scripts

python run_evolve.py --run_name fc_xdecay_zf005_sf4_xesink --zf 005 --sf 4

# 21cmFAST: 1.02