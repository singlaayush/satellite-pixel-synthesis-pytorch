DATA_PATH="/deep/group/aicc-bootcamp/transportation/data/"
TRAIN_CSV="/deep/group/aicc-bootcamp/transportation/data/texas/texas_train.csv"
TEST_CSV="/deep/group/aicc-bootcamp/transportation/data/texas/texas_test.csv"
OUTPUT_PATH="/deep/group/aicc-bootcamp/transportation/models/generative/texas"
OUTPUT_DIR="7nov22_profiled_deep"

torchrun --standalone --nnodes=1 --nproc_per_node=2 train_3dis_profiled.py --num_workers 4 --data_path $DATA_PATH --path $TRAIN_CSV --test_path $TEST_CSV --output_dir $OUTPUT_DIR --out_path $OUTPUT_PATH
