from ..modelzoo import get_config

import os
from omegaconf import OmegaConf
from unlanedet.config import LazyCall as L

from unlanedet.model.module import Detector
from unlanedet.model.laneatt import LaneATT
from unlanedet.model import ResNetWrapper

from fvcore.common.param_scheduler import CosineParamScheduler

from unlanedet.data.transform import *

featuremap_out_channel = 512 
featuremap_out_stride = 32 

num_points = 72
max_lanes = 5
sample_y=range(710, 150, -10)

train_parameters = dict(
    conf_threshold=None,
    nms_thres=15.,
    nms_topk=3000
)
test_parameters = dict(
    conf_threshold=0.2,
    nms_thres=45,
    nms_topk=max_lanes
)
ori_img_w=1280
ori_img_h=720
img_w=640 
img_h=360
cut_height=0
data_root = "/home/dataset/tusimple"

param_config = OmegaConf.create()
param_config.featuremap_out_channel = featuremap_out_channel
param_config.featuremap_out_stride = featuremap_out_stride
param_config.num_points = num_points
param_config.max_lanes = max_lanes
param_config.sample_y = list(sample_y)
param_config.ori_img_w = ori_img_w
param_config.ori_img_h = ori_img_h
param_config.img_w = img_w
param_config.img_h = img_h
param_config.cut_height = cut_height
param_config.train_parameters = train_parameters
param_config.test_parameters = test_parameters

model = L(Detector)(
    backbone = L(ResNetWrapper)(
        resnet='resnet18',
        pretrained=True,
        replace_stride_with_dilation=[False, False, False],
        out_conv=False,
    ),
    head = L(LaneATT)(
        anchors_freq_path='config/laneatt/tusimple_anchors_freq.pt',
        topk_anchors=1000,
        cfg = param_config
    )
)

train = get_config("config/common/train.py").train
epochs = 50
batch_size = 24
epoch_per_iter = (3616 // batch_size + 1)
total_iter = epoch_per_iter * epochs 
train.max_iter = total_iter
train.checkpointer.period=epoch_per_iter
train.eval_period = epoch_per_iter

optimizer = get_config("config/common/optim.py").AdamW
optimizer.lr = 0.6e-3
optimizer.weight_decay = 0.01

lr_multiplier = L(CosineParamScheduler)(
    start_value = 1,
    end_value = 0.001
)

train_process = [
    L(GenerateLaneLineATT)(
        transforms = (
            dict(
                name = 'Affine',
                parameters = dict(
                    translate_px = dict(
                        x = (-25, 25),
                        y = (-10, 10)
                    ),
                    rotate=(-6, 6),
                    scale=(0.85, 1.15),
                     
                ),
            ),
            dict(
                name = 'HorizontalFlip',
                parameters = dict(
                    p=0.5
                ),
            )
        ),
        cfg = param_config 
    ),
    L(ToTensor)(keys=['img', 'lane_line'])
] 

val_process = [
    L(GenerateLaneLineATT)(cfg = param_config),
    L(ToTensor)(keys=['img']),
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

# Evaluation config
dataloader.evaluator.output_basedir = "./output"
dataloader.evaluator.test_json_file=os.path.join(data_root,"test_label.json")