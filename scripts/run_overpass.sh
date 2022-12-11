#!/bin/sh
#SBATCH --account=deep
#SBATCH --partition=deep
#SBATCH --time=24:00:00               # Time limit hrs:min:sec
#SBATCH --output=/deep/u/ayushsn/satellite-pixel-synthesis-pytorch/logs/overpass-debug-%A.log   # Standard output and error log
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=32000

# list out some useful information (optional)
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST = "$SLURM_JOB_NODELIST
echo "SLURM_NNODES = "$SLURM_NNODES
echo "SLURMTMPDIR = "$SLURMTMPDIR
echo "Working Dir = "$SLURM_SUBMIT_DIR

python /deep/u/ayushsn/satellite-pixel-synthesis-pytorch/preprocess/overpass.py