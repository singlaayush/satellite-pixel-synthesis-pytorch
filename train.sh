mkdir /scr/data
mkdir /scr/out
rsync -a --info=progress2 --no-i-r /deep/group/aicc-bootcamp/transportation/data/texas /scr/data

DATA_PATH="/scr/data/"
TRAIN_CSV="/scr/data/texas/texas_train.csv"
TEST_CSV="/scr/data/texas/texas_test.csv"
OUTPUT_PATH="/scr/out"
OUTPUT_DIR="7nov22_scr"

python -m torch.distributed.launch --nproc_per_node=8 --master_port=1234 train_3dis.py --data_path $DATA_PATH --path $TRAIN_CSV --test_path $TEST_CSV --output_dir $OUTPUT_DIR --out_path $OUTPUT_PATH

rsync -a --info=progress2 --no-i-r /scr/out/fid/$OUTPUT_DIR /deep/group/aicc-bootcamp/transportation/models/generative/texas/fid
rsync -a --info=progress2 --no-i-r /scr/out/outputs/$OUTPUT_DIR /deep/group/aicc-bootcamp/transportation/models/generative/texas/outputs
rsync -a --info=progress2 --no-i-r /scr/out/tensorboard/$OUTPUT_DIR /deep/group/aicc-bootcamp/transportation/models/generative/texas/tensorboard

rm -rf /scr/out
rm -rf /scr/data