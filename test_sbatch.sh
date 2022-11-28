#!/bin/sh
#SBATCH --account=deep
#SBATCH --partition=deep
#SBATCH --time=47:59:59
#SBATCH --gres=gpu:1
#SBATCH --mem=32000
#SBATCH --cpus-per-task=4
#SBATCH --exclude=deep[17-25]
#SBATCH --output=/deep/u/ayushsn/satellite-pixel-synthesis-pytorch/test_sbatch.log

TEST_CSV="/deep/group/aicc-bootcamp/transportation/data/texas/texas_test.csv"
OUTPUT_PATH="/deep/group/aicc-bootcamp/transportation/models/generative/texas/test"
OUTPUT_DIR="nov22"
CKPT_ITER="490000"
CKPT_PATH="/deep/group/aicc-bootcamp/transportation/models/generative/texas/outputs/${OUTPUT_DIR}/checkpoints/${CKPT_ITER}.pt"

source ~/.bashrc
conda activate transportation
python test.py --num_workers 4 --test_path $TEST_CSV --output_dir $OUTPUT_DIR --out_path $OUTPUT_PATH --ckpt $CKPT_PATH
