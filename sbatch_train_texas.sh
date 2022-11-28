#!/bin/sh
#SBATCH --account=deep
#SBATCH --partition=deep
#SBATCH --time=125:00:00
#SBATCH --gres=gpu:2
#SBATCH --mem=32000
#SBATCH --cpus-per-task=4
#SBATCH --exclude=deep[17-25]
#SBATCH --output=/deep/u/ayushsn/satellite-pixel-synthesis-pytorch/train_sbatch.log

DATA_PATH="/deep/group/aicc-bootcamp/transportation/data/"
TRAIN_CSV="/deep/group/aicc-bootcamp/transportation/data/texas/texas_train.csv"
TEST_CSV="/deep/group/aicc-bootcamp/transportation/data/texas/texas_test.csv"
OUTPUT_PATH="/deep/group/aicc-bootcamp/transportation/models/generative/texas"
OUTPUT_DIR="28nov22"

torchrun --standalone --nnodes=1 --nproc_per_node=2 train_3dis.py --num_workers 16 --data_path $DATA_PATH --path $TRAIN_CSV --test_path $TEST_CSV --output_dir $OUTPUT_DIR --out_path $OUTPUT_PATH
