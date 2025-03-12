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
python tools/generate_seg.py --root $TUSIMPLEROOT
# python tools/generate_seg.py --root /root/paddlejob/workspace/train_data/datasets --savedir /root/paddlejob/workspace/train_data/datasets/seg_label
```
</details>

### VIL-100
<details>
<summary>VIL100 Preparation</summary>
  
Download [VIL-100](https://github.com/yujun0-0/mma-net). Unzip the data to `$VIL100ROOT`. Mkdir `data` folder

```Shell
cd $LANEDET_ROOT
mkdir -p data
ln -s $VIL100ROOT data/VIL100
```

For VIL100, the data structure should be
```Shell
/VIL100ROOT/VIL100/
├── Annotations
├── anno_txt
├── data
├── JPEGImages
└── Json
```
You may find anno_txt here [anno_txt.zip](https://drive.google.com/file/d/1SizP9p0n-x-GhHmpYNyhMPBpPQgS3enI/view?usp=drive_link)
</details>

## Inference/Demo
Run the [detect.py](../tools/detect.py) to test the model.

```Shell
cd UnLanedet
python tools/detect.py path_to_config path_to_checkpoint --img path_to_image --savedir path_to_output
# python tools/detect.py config/clrnet/resnet34_culane.py output/model_0005555.pth --img ../culane/00000.jpg --savedir output/
```

## Inference speed 
Run the [test_speed.py](../tools/test_speed.py) to test the inference speed of the model.

```Shell
cd UnLanedet
python tools/test_speed.py path_to_config path_to_checkpoint 
# python tools/test_speed.py config/clrnet/resnet34_culane.py output/model_0005555.pth
```
The result is tested under the python environment. If you want to get high speed, please refer to the TensorRT inference, which we do not support now.
