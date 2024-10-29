from ..modelzoo import get_config

import os
from functools import partial
import torch.nn as nn
from omegaconf import OmegaConf
from unlanedet.config import LazyCall as L

#import model component
from unlanedet.model import UFLD # detector
from unlanedet.model import ResNetWrapper
from unlanedet.model import LaneCls # detection head

# import learning rate schedule
from fvcore.common.param_scheduler import PolynomialDecayParamScheduler

# import dataset and transform
from unlanedet.data.transform import *

from ..modelzoo import get_config

num_classes = 4
griding_num = 200
featuremap_out_channel = 512
ori_img_h = 590 
ori_img_w = 1640 
img_h = 288
img_w = 800
cut_height=0
sample_y = range(589, 230, -20)
row_anchor = 'culane_row_anchor'
img_norm = dict(
    mean=[103.939, 116.779, 123.68],
    std=[1., 1., 1.]
)
data_root = "/home/ubuntu/zky/dataset/culane"
work_dir = "./culane"

param_config = OmegaConf.create()
param_config.img_height = img_h
param_config.img_width = img_w
param_config.cut_height = cut_height
param_config.ori_img_h = ori_img_h
param_config.ori_img_w = ori_img_w
param_config.featuremap_out_channel = featuremap_out_channel
param_config.sample_y = [i for i in range(589, 230, -20)]
param_config.num_classes = num_classes
param_config.ignore_label = 255


# model config
model = L(UFLD)(
    backbone = L(ResNetWrapper)(
        resnet='resnet18',
        pretrained=True,
        replace_stride_with_dilation=[False, False, False],
        out_conv=False,
    ),
    head = L(LaneCls)(
        dim = (griding_num + 1, 18, num_classes),
        featuremap_out_channel=featuremap_out_channel,
        griding_num=griding_num,
        sample_y = sample_y,
        ori_img_h = ori_img_h,
        ori_img_w = ori_img_w
    )
)


train = get_config("config/common/train.py").train
epochs =50
batch_size = 32
epoch_per_iter = (88880 // batch_size + 1)
total_iter = epoch_per_iter * epochs 
train.max_iter = total_iter
train.checkpointer.period=epoch_per_iter
train.eval_period = epoch_per_iter


# Learning rate config
lr_multiplier = L(PolynomialDecayParamScheduler)(
    base_value = 1,
    power = 0.9
)

# Optimizer config
optimizer = get_config("config/common/optim.py").SGD
optimizer.lr = 0.025


train_process = [
    L(RandomRotation)(degree=(-6, 6)),
    L(RandomUDoffsetLABEL)(max_offset=100),
    L(RandomUDoffsetLABEL)(max_offset=200),
    L(GenerateLaneCls)(
        row_anchor=row_anchor,
        num_cols=griding_num, 
        num_classes=num_classes
    ),
    L(Resize)(size=(img_w, img_h)),
    L(Normalize)(img_norm=img_norm),
    L(ToTensor)(keys=['img', 'cls_label']),
]

val_process = [
    L(Resize)(size=(img_w, img_h)),
    L(Normalize)(img_norm=img_norm),
    L(ToTensor)(keys=['img', 'cls_label']),
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
dataloader.evaluator.output_basedir = "./output"
dataloader.evaluator.cfg=param_config