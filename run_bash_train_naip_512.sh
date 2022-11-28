TRAIN_CSV="/deep/group/aicc-bootcamp/transportation/data/fusion/train.csv"
TEST_CSV="/deep/group/aicc-bootcamp/transportation/data/fusion/test.csv"
OUTPUT_PATH="/deep/group/aicc-bootcamp/transportation/models/generative/naip-dvrpc-10-15"
OUTPUT_DIR="28nov22-516"

torchrun --standalone --nnodes=1 --nproc_per_node=2 train_3dis_512.py --num_workers 16 --path $TRAIN_CSV --test_path $TEST_CSV --output_dir $OUTPUT_DIR --out_path $OUTPUT_PATH --iter 600000
