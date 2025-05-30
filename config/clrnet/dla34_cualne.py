from ..modelzoo import get_config

import os
from omegaconf import OmegaConf
from unlanedet.config import LazyCall as L

from unlanedet.model.CLRNet import CLRNet,CLRHead
from unlanedet.model import DLAWrapper,FPN

# import dataset and transform
from unlanedet.data.transform import *

from fvcore.common.param_scheduler import CosineParamScheduler

from ..modelzoo import get_config

iou_loss_weight = 2.
cls_loss_weight = 2.
xyt_loss_weight = 0.2
seg_loss_weight = 1.0
num_points = 72
max_lanes = 4
sample_y = range(589, 230, -20)
test_parameters = dict(conf_threshold=0.4, nms_thres=50, nms_topk=max_lanes)
ori_img_w = 1640
ori_img_h = 590
img_w = 800
img_h = 320
cut_height = 270
img_norm = dict(
    mean=[103.939, 116.779, 123.68],
    std=[1., 1., 1.]
)
ignore_label = 255
bg_weight = 0.4
featuremap_out_channel = 192
num_classes = 4 + 1
data_root = "/home/dataset/culane/"

param_config = OmegaConf.create()
param_config.iou_loss_weight = iou_loss_weight
param_config.cls_loss_weight = cls_loss_weight
param_config.xyt_loss_weight = xyt_loss_weight
param_config.seg_loss_weight = seg_loss_weight
param_config.num_points = num_points
param_config.max_lanes = max_lanes
param_config.sample_y = [i for i in range(589, 230, -20)]
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

model = L(CLRNet)(
    backbone = L(DLAWrapper)(
        dla='dla34',
        pretrained=True,   
    ),
    neck = L(FPN)(
        in_channels=[512, 1024, 2048],
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

train = get_config("config/common/train.py").train
epochs = 15
batch_size = 24
epoch_per_iter = (88880 // batch_size + 1)
total_iter = epoch_per_iter * epochs 
train.max_iter = total_iter
train.checkpointer.period=epoch_per_iter
train.eval_period = epoch_per_iter

optimizer = get_config("config/common/optim.py").AdamW
optimizer.lr = 0.6e-3
optimizer.weight_decay = 0.01

lr_multiplier = L(CosineParamScheduler)(
    start_value = 1.0,
    end_value = 0.001
)

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

dataloader = get_config("config/common/culane.py").dataloader
dataloader.train.dataset.processes = train_process
dataloader.train.dataset.data_root = data_root
dataloader.train.dataset.cut_height = cut_height
dataloader.train.dataset.cfg = param_config
dataloader.train.total_batch_size = batch_size
dataloader.test.dataset.processes = val_process
dataloader.test.dataset.data_root = data_root
dataloader.test.dataset.cut_height = cut_height
dataloader.test.dataset.cfg = param_config
dataloader.test.total_batch_size = batch_size

# Evaluation config
dataloader.evaluator.data_root = data_root
dataloader.evaluator.output_basedir = "./output"
dataloader.evaluator.cfg=param_config
