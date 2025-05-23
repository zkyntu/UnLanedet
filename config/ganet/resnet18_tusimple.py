from ..modelzoo import get_config

import os
from omegaconf import OmegaConf
from unlanedet.config import LazyCall as L

from unlanedet.model.GANet import DeformFPN,GANetHead,GaussianFocalLoss
from unlanedet.model import Detector,ResNetWrapper,TransConvEncoderModule
from unlanedet.model.module import L1Loss,SmoothL1Loss

# import dataset and transform
from unlanedet.data.transform import *
from unlanedet.data.tusimple import TusimpleCropBefore

from unlanedet.solver.lr_scheduler import WarmupParamScheduler
from fvcore.common.param_scheduler import PolynomialDecayParamScheduler


num_points = 72
max_lanes = 5
sample_y = range(710, 150, -10)
ori_img_w = 1280
ori_img_h = 720
img_h = 320
img_w = 800
cut_height = 160 
num_classes = 6 + 1
ignore_label = 255
bg_weight = 0.4
img_norm = dict(mean=[75.3, 76.6, 77.6], std=[50.5, 53.8, 54.3])
data_root = "/home/dataset/tusimple"

param_config = OmegaConf.create()
param_config.num_points = num_points
param_config.max_lanes = max_lanes
param_config.sample_y = [i for i in range(710, 150, -10)]
param_config.ori_img_w = ori_img_w
param_config.ori_img_h = ori_img_h
param_config.img_w = img_w
param_config.img_h = img_h
param_config.cut_height = cut_height
param_config.data_root = data_root
param_config.ignore_label = ignore_label
param_config.num_classes = num_classes

model = L(Detector)(
    backbone = L(ResNetWrapper)(
        resnet='resnet18',
        pretrained=True,
        replace_stride_with_dilation=[False, False, False],
        out_conv=False,        
    ),
    aggregator = L(TransConvEncoderModule)(
        attn_in_dims=[512, 64],
        attn_out_dims=[64, 64],
        strides=[1, 1],
        ratios=[4, 4],
        pos_shape=(1, 10, 25),
    ),
    neck = L(DeformFPN)(
        in_channels=[128, 256, 64],
        out_channels=64,
        start_level=0,
        num_outs=1,
        out_ids=[0],    # 1/8
        dcn_only_cls=True,),
    head = L(GANetHead)(
        in_channels=64,
        num_classes=1,
        hm_idx=0,
        loss_heatmap=L(GaussianFocalLoss)(
            alpha=2.0,
            gamma=4.0,
            reduction='mean',
            loss_weight=1.0            
        ),
        loss_kp_offset=L(L1Loss)(
            reduction='mean',
            loss_weight=1.0        
        ),
        loss_sp_offset=L(L1Loss)(
            reduction='mean',
            loss_weight=0.5            
        ),
        loss_aux=L(SmoothL1Loss)(
            beta=1.0/9.0,
            reduction='mean',
            loss_weight=0.2            
        ),
        test_cfg=dict(
            root_thr=1.0,
            kpt_thr=0.4,
            cluster_by_center_thr=4,
            hm_down_scale=8
        ),
        cfg=param_config    
    ),
)

train = get_config("config/common/train.py").train
epochs =70
batch_size = 32
epoch_per_iter = (3616 // batch_size + 1)
total_iter = epoch_per_iter * epochs 
train.max_iter = total_iter
train.checkpointer.period=epoch_per_iter
train.eval_period = epoch_per_iter

optimizer = get_config("config/common/optim.py").Adam
optimizer.lr = 1e-3
#optimizer.weight_decay = 0.01


core_lr_multiplier = L(PolynomialDecayParamScheduler)(
    base_value = 1,
    power = 0.9
)

lr_multiplier = L(WarmupParamScheduler)(
    scheduler = core_lr_multiplier,
    warmup_length=100 / total_iter,
    warmup_method='linear',
    warmup_factor=0.01,
)


train_process = [
    L(RandomHorizontalFlip)(),
    L(RandomAffine)(affine_ratio=0.7, degrees=10, translate=.1, scale=.2, shear=0.0,keys=['lanes']),
    L(Resize)(size=(img_w, img_h)),
    L(Normalize)(img_norm=img_norm),
    L(GenerateGAInfo)(
        radius=2,
        fpn_cfg=dict(
            hm_idx=0,
            fpn_down_scale=[8, 16, 32],
            sample_per_lane=[41, 21, 11],),
        norm_shape = (ori_img_h-cut_height,ori_img_w),
        cfg=param_config),
    L(ToTensor)(keys=['img'],collect_keys=['gt_hm_lanes', 'gt_kpts_hm','gt_kp_offset','gt_sp_offset','kp_offset_mask','sp_offset_mask']),
]

val_process = [
    L(Resize)(size=(img_w, img_h)),
    L(Normalize)(img_norm=img_norm),
    L(ToTensor)(keys=['img'])
]

dataloader = get_config("config/common/tusimple.py").dataloader
dataloader.train.dataset = L(TusimpleCropBefore)(
                                data_root = data_root,
                                split='trainval',
                                cut_height=cut_height,
                                processes=train_process,
                                cfg=param_config
                            )
dataloader.train.total_batch_size = batch_size
dataloader.test.dataset = L(TusimpleCropBefore)(
                                data_root = data_root,
                                split='test',
                                cut_height=cut_height,
                                processes=val_process,
                                cfg=param_config
                            )
dataloader.test.total_batch_size = batch_size

# Evaluation config
dataloader.evaluator.output_basedir = "./output"
dataloader.evaluator.test_json_file=os.path.join(data_root,"test_label.json")
