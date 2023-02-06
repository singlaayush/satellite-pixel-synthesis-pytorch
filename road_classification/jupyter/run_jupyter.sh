#!/bin/sh
#SBATCH --account=deep
#SBATCH --partition=deep
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32000
#SBATCH --cpus-per-task=4
#SBATCH --exclude=deep[17-25]
#SBATCH --output=/deep/u/ayushsn/satellite-pixel-synthesis-pytorch/road_classification/jupyter/logs/jupyter_%A.log

cd /deep/u/ayushsn/satellite-pixel-synthesis-pytorch/road_classification/jupyter
export XDG_RUNTIME_DIR=""
jupyter-notebook --no-browser --port=8880 --ip='0.0.0.0'
