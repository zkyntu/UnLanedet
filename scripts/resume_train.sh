CONFIG=$1
GPUS=$2

python tools/train_net.py --config-file $CONFIG --num-gpus $GPUS --resume
