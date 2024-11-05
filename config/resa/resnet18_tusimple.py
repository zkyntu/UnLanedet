import os
from omegaconf import OmegaConf
from unlanedet.config import LazyCall as L

from unlanedet.model import RESA,resa,BUSD,LaneSeg
from unlanedet.model import ResNetWrapper
from fvcore.common.param_scheduler import PolynomialDecayParamScheduler
from unlanedet.data.transform import *

from ..modelzoo import get_config

featuremap_out_channel = 128
featuremap_out_stride = 8
sample_y=range(710, 150, -10)
bg_weight = 0.4
img_norm = dict(
    mean=[103.939, 116.779, 123.68],
    std=[1., 1., 1.]
)
img_height = 368
img_width = 640
cut_height = 160
ori_img_h = 720
ori_img_w = 1280
ignore_label = 255
num_classes = 6 + 1
data_root = "/home/dataset/tusimple"

param_config = OmegaConf.create()
param_config.img_height = img_height
param_config.img_width = img_width
param_config.cut_height = cut_height
param_config.ori_img_h = ori_img_h
param_config.ori_img_w = ori_img_w
param_config.featuremap_out_channel = featuremap_out_channel
param_config.featuremap_out_stride = featuremap_out_stride
param_config.bg_weight = bg_weight
param_config.sample_y = [i for i in range(710, 150, -10)]
param_config.num_classes = num_classes
param_config.ignore_label = ignore_label
param_config.direction = ['d', 'u', 'r', 'l']

model = L(RESA)(
    backbone = L(ResNetWrapper)(
        resnet='resnet18',
        pretrained=True,
        replace_stride_with_dilation=[False, True, True],
        out_conv=True,  
        cfg=param_config,   
    ),
    aggregator = L(resa)(
        direction=['d', 'u', 'r', 'l'],
        alpha=2.0,
        iter=4,
        conv_stride=9,
        cfg = param_config
    ),
    head = L(LaneSeg)(
        decoder=L(BUSD)(cfg=param_config),
        thr=0.6,
        sample_y=sample_y,
        cfg = param_config
    )
)

train = get_config("config/common/train.py").train
epochs = 100
batch_size = 16
epoch_per_iter = (3616 // batch_size + 1)
total_iter = epoch_per_iter * epochs 
train.max_iter = total_iter
train.checkpointer.period=epoch_per_iter
train.eval_period = epoch_per_iter

lr_multiplier = L(PolynomialDecayParamScheduler)(
    base_value = 1,
    power = 0.9
)

optimizer = get_config("config/common/optim.py").SGD
optimizer.lr = 0.025

train_process = [
    L(RandomRotation)(),
    L(RandomHorizontalFlip)(),
    L(Resize)(size=(img_width, img_height)),
    L(Normalize)(img_norm=img_norm),
    L(ToTensor)()
]

val_process = [
    L(Resize)(size=(img_width, img_height)),
    L(Normalize)(img_norm=img_norm),
    L(ToTensor)(keys=['img'])
]

dataloader = get_config("config/common/tusimple.py").dataloader
dataloader.train.dataset.processes = train_process
dataloader.train.dataset.data_root = data_root
dataloader.train.dataset.cut_height = cut_height
dataloader.train.total_batch_size = batch_size
dataloader.test.dataset.processes = val_process
dataloader.test.dataset.data_root = data_root
dataloader.test.dataset.cut_height = cut_height
dataloader.test.total_batch_size = batch_size

dataloader.evaluator.output_basedir = "./output"
dataloader.evaluator.test_json_file=os.path.join(data_root,"test_label.json")