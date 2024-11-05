CONFIG=$1
GPUS=$2

python tools/train_net.py --config-file $CONFIG --num-gpus $GPUS

#multi machine training
# NM=$3 The number of the machines
# python tools/train_net.py --config-file $CONFIG --num-gpus $GPUS --num-machines $NM

# For more training codes, refer to unlanedet\engine\defaults.py