#!/bin/sh
#SBATCH --account=deep
#SBATCH --partition=deep
#SBATCH --time=125:00:00
#SBATCH --gres=gpu:2
#SBATCH --mem=32000
#SBATCH --cpus-per-task=4
#SBATCH --exclude=deep[17-25]
#SBATCH --output=/deep/u/ayushsn/satellite-pixel-synthesis-pytorch/train_naip_512.log

# list out some useful information (optional)
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST = "$SLURM_JOB_NODELIST
echo "SLURM_NNODES = "$SLURM_NNODES
echo "SLURMTMPDIR = "$SLURMTMPDIR
echo "Working Dir = "$SLURM_SUBMIT_DIR

DATA_PATH=""
TRAIN_CSV="/deep/group/aicc-bootcamp/transportation/data/fusion/train.csv"
TEST_CSV="/deep/group/aicc-bootcamp/transportation/data/fusion/test.csv"
OUTPUT_PATH="/deep/group/aicc-bootcamp/transportation/models/generative/naip-dvrpc-10-15"
OUTPUT_DIR="28nov22-512"

torchrun --standalone --nnodes=1 --nproc_per_node=2 train_3dis_512.py --num_workers 16 --data_path $DATA_PATH --path $TRAIN_CSV --test_path $TEST_CSV --output_dir $OUTPUT_DIR --out_path $OUTPUT_PATH --coords_size 512 --iter 600000
