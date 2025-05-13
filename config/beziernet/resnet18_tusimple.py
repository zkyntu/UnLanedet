from ..modelzoo import get_config

import os
from omegaconf import OmegaConf
from unlanedet.config import LazyCall as L

from torch.optim.lr_scheduler import CosineAnnealingLR
from unlanedet.solver.build import get_default_optimizer_params

from unlanedet.model import ResNetWrapper
from unlanedet.model.Beizernet import BezierHead,BezierHungarianAssigner,BezierLaneNet,FeatureFlipFusion
from unlanedet.model import CrossEntropyLoss,L1Loss

from unlanedet.data.tusimple import BiazerTusimple

from unlanedet.data.transform import *

from unlanedet.solver.lr_scheduler import WarmupParamScheduler
from fvcore.common.param_scheduler import CosineParamScheduler

order = 3
img_w = 640
img_h = 320
ori_img_w = 1280
ori_img_h = 720
cut_height = 160
img_norm = dict(
    mean=[103.939, 116.779, 123.68],
    std=[1., 1., 1.]
)
img_shape = (img_h,img_w)
ori_img_shape = (ori_img_h,ori_img_w)
window_size = 9
data_root = "/home/zky/dataset/tusimple"

param_config = OmegaConf.create()
param_config.order = order
param_config.img_shape = img_shape
param_config.ori_img_shape = ori_img_shape
param_config.cut_height = cut_height
param_config.dataset_type = "tusimple"


model = L(BezierLaneNet)(
    backbone=L(ResNetWrapper)(
        resnet='resnet18',
        pretrained=True,
        replace_stride_with_dilation=[False, False, False],
        out_conv=False,            
    ),
    dilated_blocks=dict(
        in_channels=256,
        mid_channels=64,
        dilations=[4, 8]
    ),
    aggregator=L(FeatureFlipFusion)(
        channels=256
    ),
    head=L(BezierHead)(
        in_channels=256,
        branch_channels=256,
        num_proj_layers=2,
        feature_size=(20, 40),
        order=order,
        with_seg=False,
        num_classes=1,
        seg_num_classes=1,  
        loss_cls=L(CrossEntropyLoss)(
            use_sigmoid=True,
            class_weight=1.0 / 0.4,
            reduction='mean',
            loss_weight=0.1
        ),      
        loss_reg=L(L1Loss)(
            reduction='mean',
            loss_weight=1.0,
        ),
        loss_seg=L(CrossEntropyLoss)(
            use_sigmoid=True,
            ignore_index=255,
            class_weight=1.0 / 0.4,
            loss_weight=0.75,
            reduction='mean',
        ),
        assigner=L(BezierHungarianAssigner)(
            order=order,
            num_sample_points=100,
            alpha=0.8,
            window_size=window_size       
        ),
        test_cfg=dict(
            score_thr=0.4,
            window_size=window_size,
            max_lanes=5,
            num_sample_points=50,
            dataset='tusimple'
        ),
        cfg=param_config
    ),
)

train = get_config("config/common/train.py").train
epochs = 400
batch_size = 32
epoch_per_iter = (3616 // batch_size + 1)
total_iter = epoch_per_iter * epochs 
train.max_iter = total_iter
train.checkpointer.period=epoch_per_iter
train.eval_period = epoch_per_iter * 5

optimizer = get_config("config/common/optim.py").Adam
optimizer.lr = 1e-3
optimizer.params.lr_factor_func = lambda module_name: 0.1 if "conv_offset" in module_name else 1

core_lr_multiplier = L(CosineParamScheduler)(
    start_value = 1,
    end_value = 0.001
)

lr_multiplier = L(WarmupParamScheduler)(
    scheduler = core_lr_multiplier,
    warmup_length=500 / total_iter,
    warmup_method='linear',
    warmup_factor=0.001,
)

train_process = [
    L(RandomHorizontalFlip)(),
    L(RandomAffine)(affine_ratio=0.7, degrees=10, translate=0.1, scale=0.2, shear=0.0),
    L(Resize)(size=(img_w, img_h)),
    L(Normalize)(img_norm=img_norm),
    L(GenerateBezierInfo)(order=order, num_sample_points=100,cfg=param_config),
    L(ListToTensor)(keys=['img', 'gt_control_points', 'lanes_labels', 'gt_semantic_seg'])
]

val_process = [
    L(Resize)(size=(img_w, img_h)),
    L(Normalize)(img_norm=img_norm),
    L(ToTensor)(keys=['img'])
]


dataloader = get_config("config/common/tusimple.py").dataloader
dataloader.train.dataset = L(BiazerTusimple)(
                                data_root = "./tusimple",
                                split='trainval',
                                cut_height=cut_height,
                                processes=None,
                                cfg=param_config
                            )
dataloader.train.dataset.processes = train_process
dataloader.train.dataset.data_root = data_root
dataloader.train.dataset.cut_height = cut_height
dataloader.train.total_batch_size = batch_size
dataloader.test.dataset = L(BiazerTusimple)(
                                data_root = "./tusimple",
                                split='test',
                                cut_height=cut_height,
                                processes=None,
                                cfg=param_config
                            )
dataloader.test.dataset.processes = val_process
dataloader.test.dataset.data_root = data_root
dataloader.test.dataset.cut_height = cut_height
dataloader.test.total_batch_size = batch_size

dataloader.evaluator.output_basedir = "./output"
dataloader.evaluator.test_json_file=os.path.join(data_root,"test_label.json")