from ..modelzoo import get_config

import os
from omegaconf import OmegaConf
from unlanedet.config import LazyCall as L

from unlanedet.model.ADNet import SA_FPN,SPGHead
from unlanedet.model import ResNetWrapper,Detector

from unlanedet.data.transform import *

from fvcore.common.param_scheduler import CosineParamScheduler

in_channels= [128,256,512]
anchor_feat_channels = 64
num_points = 72
# basic setting
img_w = 800
img_h = 320
# network setting
fpn_down_scale = [8,16,32]
anchors_num = 100
regw = 10
hmw = 10
thetalossw = 1
cls_loss_w = 10
dynamic_after = 60
do_mask = False
max_lanes = 6
train_parameters = dict(
    conf_threshold=None,
    nms_thres=45.,
    nms_topk=max_lanes
)
test_parameters = dict(
    conf_threshold=0.3,
    nms_thres=45,
    nms_topk=max_lanes
)
sample_y=range(1080, 134, -1)
ori_img_w=1280
ori_img_h=720
cut_height=160
hm_down_scale = 8
keys = ['img', 'lane_line','gt_hm','shape_hm','shape_hm_mask']
data_root = "/home/zky/dataset/VIL100/VIL100"

param_config = OmegaConf.create()
param_config.in_channels = in_channels
param_config.anchor_feat_channels = anchor_feat_channels
param_config.num_points = num_points
param_config.img_w = img_w
param_config.img_h = img_h
param_config.fpn_down_scale = fpn_down_scale
param_config.anchors_num = anchors_num
param_config.regw = regw
param_config.hmw = hmw
param_config.thetalossw = thetalossw
param_config.cls_loss_w = cls_loss_w
param_config.dynamic_after = dynamic_after
param_config.do_mask = do_mask
param_config.max_lanes = max_lanes
param_config.train_parameters = train_parameters
param_config.test_parameters = test_parameters
param_config.sample_y = list(sample_y)
param_config.ori_img_w = ori_img_w
param_config.ori_img_h = ori_img_h
param_config.cut_height = cut_height
param_config.hm_down_scale = hm_down_scale

model = L(Detector)(
    backbone = L(ResNetWrapper)(
        resnet='resnet34',
        pretrained=True,
        replace_stride_with_dilation=[False, False, False],
        out_conv=False,
        ),
    neck = L(SA_FPN)(
            in_channels=in_channels,
            out_channels=anchor_feat_channels,
            num_outs=len(in_channels)
            ),
    head = L(SPGHead)(
        S = num_points,
        anchor_feat_channels = anchor_feat_channels,
        img_width = img_w,
        img_height = img_h,
        start_points_num=anchors_num,
        cfg = param_config       
    )
)

train = get_config("config/common/train.py").train
epochs = 80
batch_size = 45
epoch_per_iter = (8000 // batch_size + 1)
total_iter = epoch_per_iter * epochs 
train.max_iter = total_iter
train.checkpointer.period=epoch_per_iter
train.eval_period = epoch_per_iter
train.output_dir = "output_vil"

optimizer = get_config("config/common/optim.py").Adam
optimizer.lr =  0.0012

lr_multiplier = L(CosineParamScheduler)(
    start_value = 1,
    end_value = 0.001
)

train_process = [
    L(GenerateLanePts)(
        transforms=[
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
        cfg=param_config,      
    ),
    L(CollectHm)(
        down_scale=hm_down_scale,
        hm_down_scale=hm_down_scale,
        max_mask_sample=5,
        line_width=3,
        radius=6,
        theta_thr = 0.2,
        keys=keys,
        meta_keys=['gt_points'],
        cfg=param_config        
    ),
    L(ToTensor)(keys=keys),
]

val_process = [
    L(GenerateLanePts)(
        training = False,
        transforms=[
            dict(name='Resize',
                 parameters=dict(size=dict(height=img_h, width=img_w)),
                 p=1.0),
        ],
        cfg=param_config      
    ),
    L(ToTensor)(keys=['img','lane_line']),
]

dataloader = get_config("config/common/vil.py").dataloader
dataloader.train.dataset.processes = train_process
dataloader.train.dataset.data_root = data_root
dataloader.train.dataset.cut_height = cut_height
dataloader.train.total_batch_size = batch_size
dataloader.test.dataset.processes = val_process
dataloader.test.dataset.data_root = data_root
dataloader.test.dataset.cut_height = cut_height
dataloader.test.total_batch_size = batch_size

# Evaluation config
dataloader.evaluator.output_basedir = "./output_vil"
dataloader.evaluator.data_root = data_root
