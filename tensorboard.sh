#!/bin/sh
#SBATCH --account=deep
#SBATCH --partition=deep
#SBATCH --ntasks=1                    # Run on a single CPU
#SBATCH --time=02:00:00               # Time limit hrs:min:sec
#SBATCH --output=/deep/u/ayushsn/satellite-pixel-synthesis-pytorch/tb1.log   # Standard output and error log
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G

export XDG_RUNTIME_DIR=""
tensorboard --logdir=/deep/group/aicc-bootcamp/transportation/models/generative/texas/tensorboard --port=8881 --host='0.0.0.0'