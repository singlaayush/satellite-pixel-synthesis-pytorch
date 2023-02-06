#!/bin/sh
#SBATCH --account=deep
#SBATCH --partition=deep
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:4
#SBATCH --mem=32000
#SBATCH --cpus-per-task=4
#SBATCH --exclude=deep[17-25]
#SBATCH --output=/deep/u/ayushsn/satellite-pixel-synthesis-pytorch/road_classification/logs/version_%A_validate.log

cd /deep/u/ayushsn/satellite-pixel-synthesis-pytorch/road_classification
srun python flash_validate.py