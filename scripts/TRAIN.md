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
Currently, UnLanedet achieves Tusimple, CULane dataset, and VIL100 dataset.

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

## Advanced Usage
<details>
<summary>Adding model to UnLanedet</summary>
We introduce how to add the new model to UnLanedet. We take CLRNet as an example to describe this process.

1. Create the folder for the model under ```unlanedet/model```, such as CLRNet.

2. Add the core model under CLRNet folder.

3. create the config file. Following the example config file below.

```Shell
from ..modelzoo import get_config

import os
from omegaconf import OmegaConf
from unlanedet.config import LazyCall as L
#==the above modules are general==

#=========import the model============
from unlanedet.model.CLRNet import CLRNet,CLRHead
from unlanedet.model import ResNetWrapper,FPN


# import dataset and transform
from unlanedet.data.transform import *

# import learning schedule
from fvcore.common.param_scheduler import CosineParamScheduler

# parameter setting (the necessary parameter for your model)
iou_loss_weight = 2.
cls_loss_weight = 6.
xyt_loss_weight = 0.5
seg_loss_weight = 1.0
num_points = 72
max_lanes = 5
sample_y = range(710, 150, -10)
img_norm = dict(mean=[103.939, 116.779, 123.68], std=[1., 1., 1.])
ori_img_w = 1280
ori_img_h = 720
img_h = 320
img_w = 800
cut_height = 160 
num_classes = 6 + 1
ignore_label = 255
bg_weight = 0.4
featuremap_out_channel = 192
test_parameters = dict(conf_threshold=0.4, nms_thres=50, nms_topk=max_lanes)

# dataset path
data_root = "/home/dataset/tusimple"

# Wrapper the parameter, which is also general
param_config = OmegaConf.create()
param_config.iou_loss_weight = iou_loss_weight
param_config.cls_loss_weight = cls_loss_weight
param_config.xyt_loss_weight = xyt_loss_weight
param_config.seg_loss_weight = seg_loss_weight
param_config.num_points = num_points
param_config.max_lanes = max_lanes
param_config.sample_y = [i for i in range(710, 150, -10)]
param_config.test_parameters = test_parameters
param_config.ori_img_w = ori_img_w
param_config.ori_img_h = ori_img_h
param_config.img_w = img_w
param_config.img_h = img_h
param_config.cut_height = cut_height
param_config.img_norm = img_norm
param_config.data_root = data_root
param_config.ignore_label = ignore_label
param_config.bg_weight = bg_weight
param_config.featuremap_out_channel = featuremap_out_channel
param_config.num_classes = num_classes

# Create the model
model = L(CLRNet)(
    backbone = L(ResNetWrapper)(
        resnet='resnet34',
        pretrained=True,
        replace_stride_with_dilation=[False, False, False],
        out_conv=False,        
    ),
    neck = L(FPN)(
        in_channels=[128, 256, 512],
        out_channels=64,
        num_outs=3,
        attention=False),
    head = L(CLRHead)(
        num_priors=192,
        refine_layers=3,
        fc_hidden_dim=64,
        sample_points=36,
        cfg=param_config
    )
)

# Create the training program, including batch size and the number of training iters
train = get_config("config/common/train.py").train
epochs =70
batch_size = 32
epoch_per_iter = (3616 // batch_size + 1)
total_iter = epoch_per_iter * epochs 
train.max_iter = total_iter
train.checkpointer.period=epoch_per_iter
train.eval_period = epoch_per_iter

# create the optimizer
optimizer = get_config("config/common/optim.py").AdamW
optimizer.lr = 0.8e-3
optimizer.weight_decay = 0.01

# create the learning schedule
lr_multiplier = L(CosineParamScheduler)(
    start_value = 1,
    end_value = 0.001
)

# create the data preprocess
train_process = [
    L(GenerateLaneLine)(
        transforms = [
            dict(name='Resize',
                 parameters=dict(size=dict(height=img_h, width=img_w)),
                 p=1.0),
            dict(name='HorizontalFlip', parameters=dict(p=1.0), p=0.5),
            dict(name='ChannelShuffle', parameters=dict(p=1.0), p=0.1),
            dict(name='MultiplyAndAddToBrightness',
                 parameters=dict(mul=(0.85, 1.15), add=(-10, 10)),
                 p=0.6),
            dict(name='AddToHueAndSaturation',
                 parameters=dict(value=(-10, 10)),
                 p=0.7),
            dict(name='OneOf',
                 transforms=[
                     dict(name='MotionBlur', parameters=dict(k=(3, 5))),
                     dict(name='MedianBlur', parameters=dict(k=(3, 5)))
                 ],
                 p=0.2),
            dict(name='Affine',
                 parameters=dict(translate_percent=dict(x=(-0.1, 0.1),
                                                        y=(-0.1, 0.1)),
                                 rotate=(-10, 10),
                                 scale=(0.8, 1.2)),
                 p=0.7),
            dict(name='Resize',
                 parameters=dict(size=dict(height=img_h, width=img_w)),
                 p=1.0),            
        ],
        cfg = param_config
    ),
    L(ToTensor)(keys=['img', 'lane_line', 'seg']),
]

val_process = [
    L(GenerateLaneLine)(
         transforms=[
             dict(name='Resize',
                  parameters=dict(size=dict(height=img_h, width=img_w)),
                  p=1.0),
         ],
         training=False,
         cfg = param_config        
    ),
    L(ToTensor)(keys=['img'])
]

# create the dataloader
dataloader = get_config("config/common/tusimple.py").dataloader
dataloader.train.dataset.processes = train_process
dataloader.train.dataset.data_root = data_root
dataloader.train.dataset.cut_height = cut_height
dataloader.train.total_batch_size = batch_size
dataloader.test.dataset.processes = val_process
dataloader.test.dataset.data_root = data_root
dataloader.test.dataset.cut_height = cut_height
dataloader.test.total_batch_size = batch_size

# Evaluation config
dataloader.evaluator.output_basedir = "./output"
dataloader.evaluator.test_json_file=os.path.join(data_root,"test_label.json")

```

4. Run the training code.

Note: UnLanedet is built on lazy configuration. Therefore, UnLanedet does not require the registry for the model, just importing your model in the config file.
</details>

