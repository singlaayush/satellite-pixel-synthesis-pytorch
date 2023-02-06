#!/bin/sh
#SBATCH --account=deep
#SBATCH --partition=deep
#SBATCH --time=125:00:00
#SBATCH --gres=gpu:2
#SBATCH --mem=32000
#SBATCH --cpus-per-task=4
#SBATCH --exclude=deep[17-25]
#SBATCH --output=/deep/u/ayushsn/satellite-pixel-synthesis-pytorch/logs/train_vis_3dec22_centercrop_d_2_lr_0.004_%A.log
CAPACITY_MULTIPLIER=2

# list out some useful information (optional)
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST = "$SLURM_JOB_NODELIST
echo "SLURM_NNODES = "$SLURM_NNODES
echo "SLURMTMPDIR = "$SLURMTMPDIR
echo "Working Dir = "$SLURM_SUBMIT_DIR

TRAIN_CSV="/deep/group/aicc-bootcamp/transportation/data/fusion/train.csv"
TEST_CSV="/deep/group/aicc-bootcamp/transportation/data/fusion/test_vis.csv"
OUTPUT_PATH="/deep/group/aicc-bootcamp/transportation/models/generative/naip-dvrpc-10-15"
OUTPUT_DIR="12dec22_vis_hardcoded"
CKPT="/deep/group/aicc-bootcamp/transportation/models/generative/naip-dvrpc-10-15/outputs/29nov22_centercrop_lr_0.004_d_2/checkpoints/360000.pt"
#OUTPUT_DIR="29nov22_lr_0.052"

torchrun --standalone --nnodes=1 --nproc_per_node=2 train_3dis.py --num_workers 16 --path $TRAIN_CSV --test_path $TEST_CSV --output_dir $OUTPUT_DIR --out_path $OUTPUT_PATH --capacity_multiplier $CAPACITY_MULTIPLIER --lr 0.004 --iter 600000
#torchrun --standalone --nnodes=1 --nproc_per_node=2 train_3dis.py --num_workers 16 --path $TRAIN_CSV --test_path $TEST_CSV --output_dir $OUTPUT_DIR --out_path $OUTPUT_PATH --ckpt $CKPT --capacity_multiplier $CAPACITY_MULTIPLIER --lr 0.004 --iter 600000
