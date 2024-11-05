CONFIG=$1
CHECKPOINT=$2

python tools/train_net.py --config-file $CONFIG --eval-only train.init_checkpoint=$CHECKPOINT
