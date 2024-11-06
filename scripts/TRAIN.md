## Engine code

### Training
```Shell
cd UnLanedet
bash scripts/train.sh path_to_config num_gpus
# 8 cards training example 
# bash scripts/train.sh config/clrnet/resnet34_culane.py 8
```

### Evaluation
```Shell
cd UnLanedet
bash scripts/eval.sh path_to_config path_to_checkpoint
# 8 cards training example 
# bash scripts/eval.sh config/clrnet/resnet34_culane.py model_1000.path
```

### Resume
```Shell
cd UnLanedet
bash scripts/resume_train.sh path_to_config num_gpus
# 8 cards training example 
# bash scripts/resume_train.sh config/clrnet/resnet34_culane.py 8
```

## Data Preparation
Currently, UnLanedet achieves Tusimple and CULane dataset. In the future, we will support more lane detection datasets.

### CULane
<details>
<summary>CULane Preparation</summary>

Download [CULane](https://xingangpan.github.io/projects/CULane.html). Unzip the data to `$CULANEROOT`. Mkdir `data` folder.

```Shell
cd $LANEDET_ROOT
mkdir -p data
ln -s $CULANEROOT data/CULane
```

For CULane, the data structure should be
```
$CULANEROOT/driver_xx_xxframe    # data folders x6
$CULANEROOT/laneseg_label_w16    # lane segmentation labels
$CULANEROOT/list                 # data lists
```
</details>

### Tusimple
<details>
<summary>Tusimple Preparation</summary>
  
Download [Tusimple](https://github.com/TuSimple/tusimple-benchmark/issues/3). Unzip the data to `$TUSIMPLEROOT`. Mkdir `data` folder

```Shell
cd $LANEDET_ROOT
mkdir -p data
ln -s $TUSIMPLEROOT data/tusimple
```

For Tusimple, the data structure should be
```
$TUSIMPLEROOT/clips # data folders
$TUSIMPLEROOT/lable_data_xxxx.json # label json file x4
$TUSIMPLEROOT/test_tasks_0627.json # test tasks json file
$TUSIMPLEROOT/test_label.json # test label json file

```

Tusimple does not provide segmentation label. You can run the following code to gengerate the segmentation mask. 

```Shell
python tools/generate_seg_tusimple.py --root $TUSIMPLEROOT
# python tools/generate_seg_tusimple.py --root /root/paddlejob/workspace/train_data/datasets --savedir /root/paddlejob/workspace/train_data/datasets/seg_label
```
</details>

