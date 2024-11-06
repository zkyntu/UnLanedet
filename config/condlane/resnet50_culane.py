from ..modelzoo import get_config

import os
from omegaconf import OmegaConf
from unlanedet.config import LazyCall as L

from unlanedet.model import ResNetWrapper,FPN,TransConvEncoderModule
from unlanedet.model.CondlaneNet import CondLaneHead,CtnetHead,CondLaneNet

from unlanedet.data.transform import *

from fvcore.common.param_scheduler import MultiStepParamScheduler

num_lane_classes=1
sample_y = range(590, 270, -8)
img_norm = dict(
    mean=[75.3, 76.6, 77.6],
    std=[50.5, 53.8, 54.3]
)
img_height = 320 
img_width = 800
cut_height = 0 
ori_img_h = 590
ori_img_w = 1640
mask_down_scale = 4
hm_down_scale = 16
num_lane_classes = 1
line_width = 3
radius = 6
nms_thr = 4
img_scale = (800, 320)
crop_bbox = [0, 270, 1640, 590]
mask_size = (1, 80, 200)
seg_loss_weight = 1.0
loss_weights=dict(
        hm_weight=1,
        kps_weight=0.4,
        row_weight=1.,
        range_weight=1.,
    )
data_root = "/home/dataset/culane"

param_config = OmegaConf.create()
param_config.num_lane_classes = num_lane_classes
param_config.sample_y = list(sample_y)
param_config.img_norm = img_norm
param_config.img_w = img_width
param_config.img_h = img_height
param_config.ori_img_h = ori_img_h
param_config.ori_img_w = ori_img_w
param_config.mask_down_scale = mask_down_scale
param_config.hm_down_scale = hm_down_scale
param_config.num_lane_classes = num_lane_classes
param_config.line_width = line_width
param_config.radius = radius
param_config.nms_thr = nms_thr
param_config.img_scale = img_scale
param_config.crop_bbox = crop_bbox
param_config.mask_size = mask_size
param_config.seg_loss_weight = seg_loss_weight
param_config.loss_weights = loss_weights

train = get_config("config/common/train.py").train
epochs = 15
batch_size = 8
epoch_per_iter = (88880 // batch_size + 1)
total_iter = epoch_per_iter * epochs 
train.max_iter = total_iter
train.checkpointer.period=epoch_per_iter
train.eval_period = epoch_per_iter

model = L(CondLaneNet)(
    backbone = L(ResNetWrapper)(
        resnet='resnet50',
        pretrained=True,
        replace_stride_with_dilation=[False, False, False],
        out_conv=False,
        in_channels=[64, 128, 256, 512]        
    ),
    aggregator = L(TransConvEncoderModule)(
        in_dim=2048,
        attn_in_dims=[2048, 256],
        attn_out_dims=[256, 256],
        strides=[1, 1],
        ratios=[4, 4],
        pos_shape=(batch_size, 10, 25),
    ),
    neck=L(FPN)(
        in_channels=[256, 512, 1024, 256],
        out_channels=64,
        num_outs=4,
        #trans_idx=-1,
    ),
    head = L(CondLaneHead)(
        heads=dict(hm=num_lane_classes),
        in_channels=(64, ),
        num_classes=num_lane_classes,
        head_channels=64,
        head_layers=1,
        disable_coords=False,
        branch_in_channels=64,
        branch_channels=64,
        branch_out_channels=64,
        reg_branch_channels=64,
        branch_num_conv=1,
        hm_idx=2,
        mask_idx=0,
        compute_locations_pre=True,
        location_configs=dict(size=(batch_size, 1, 80, 200), device='cuda:0'),
        cfg = param_config     
    )
)

optimizer = get_config("config/common/optim.py").AdamW
optimizer.lr = 0.6e-3
optimizer.weight_decay = 0.01

lr_multiplier = L(MultiStepParamScheduler)(
    values=[0.1, 0.01],
    milestones=[8, 14]
)

train_process = [
    L(Alaug)(
        transforms=[dict(type='Compose', params=dict(bboxes=False, keypoints=True, masks=False)),
        dict(
            type='Crop',
            x_min=crop_bbox[0],
            x_max=crop_bbox[2],
            y_min=crop_bbox[1],
            y_max=crop_bbox[3],
            p=1),
        dict(type='Resize', height=img_scale[1], width=img_scale[0], p=1),
        dict(
            type='OneOf',
            transforms=[
                dict(
                    type='RGBShift',
                    r_shift_limit=10,
                    g_shift_limit=10,
                    b_shift_limit=10,
                    p=1.0),
                dict(
                    type='HueSaturationValue',
                    hue_shift_limit=(-10, 10),
                    sat_shift_limit=(-15, 15),
                    val_shift_limit=(-10, 10),
                    p=1.0),
            ],
            p=0.7),
        dict(type='JpegCompression', quality_lower=85, quality_upper=95, p=0.2),
        dict(
            type='OneOf',
            transforms=[
                dict(type='Blur', blur_limit=3, p=1.0),
                dict(type='MedianBlur', blur_limit=3, p=1.0)
            ],
            p=0.2),
        dict(type='RandomBrightness', limit=0.2, p=0.6),
        dict(
            type='ShiftScaleRotate',
            shift_limit=0.1,
            scale_limit=(-0.2, 0.2),
            rotate_limit=10,
            border_mode=0,
            p=0.6),
        dict(
            type='RandomResizedCrop',
            height=img_scale[1],
            width=img_scale[0],
            scale=(0.8, 1.2),
            ratio=(1.7, 2.7),
            p=0.6),
        dict(type='Resize', height=img_scale[1], width=img_scale[0], p=1),]
        ),
    L(CollectLane)(
        down_scale=mask_down_scale,
        hm_down_scale=hm_down_scale,
        max_mask_sample=5,
        line_width=line_width,
        radius=radius,
        keys=['img', 'gt_hm'],
        meta_keys=[
            'gt_masks', 'mask_shape', 'hm_shape',
            'down_scale', 'hm_down_scale', 'gt_points'                    
        ],
        cfg = param_config             
    ),
    L(Normalize)(img_norm=img_norm),
    L(ToTensor)(keys=['img', 'gt_hm'], collect_keys=['img_metas'])
]

val_process = [
    L(Alaug)(
        transforms=[dict(type='Compose', params=dict(bboxes=False, keypoints=True, masks=False)),
            dict(type='Crop',
            x_min=crop_bbox[0],
            x_max=crop_bbox[2],
            y_min=crop_bbox[1],
            y_max=crop_bbox[3],
            p=1),
        dict(type='Resize', height=img_scale[1], width=img_scale[0], p=1)]
    ),
    #dict(type='Resize', size=(img_width, img_height)),
    L(Normalize)(img_norm=img_norm),
    L(ToTensor)(keys=['img'])
]

dataloader = get_config("config/common/culane.py").dataloader
dataloader.train.dataset.processes = train_process
dataloader.train.dataset.data_root = data_root
dataloader.train.dataset.cfg = param_config
dataloader.train.total_batch_size = batch_size
dataloader.test.dataset.processes = val_process
dataloader.test.dataset.data_root = data_root
dataloader.test.dataset.cfg = param_config
dataloader.test.total_batch_size = batch_size

# Evaluation config
dataloader.evaluator.data_root = data_root
dataloader.evaluator.output_basedir = "./output"
dataloader.evaluator.cfg=param_config