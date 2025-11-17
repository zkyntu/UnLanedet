from ..modelzoo import get_config

import os
from omegaconf import OmegaConf
from unlanedet.config import LazyCall as L

from unlanedet.model.DiffusionLane import DiffLane,DiffusionLaneHead,AUX_DiffLane,AUXCLRHead
from unlanedet.model import ResNetWrapper,FPN
from unlanedet.model.ADNet import  SA_FPN
from unlanedet.model.CLRNet import CLRHead

# import dataset and transform
from unlanedet.data.transform import *

from fvcore.common.param_scheduler import CosineParamScheduler

from ..modelzoo import get_config

from ..clrnet.resnet34_tusimple import param_config

iou_loss_weight = 2.
cls_loss_weight = 6.
xyt_loss_weight = 0.5
seg_loss_weight = 1.0
df_loss_weight = 1.
angle_loss_weight = 0.02
num_points = 72
max_lanes = 5
sample_y = range(710, 150, -10)
img_norm = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
ori_img_w = 1280
ori_img_h = 720
img_h = 320
img_w = 800
cut_height = 160
num_classes = 6 + 1
ignore_label = 255
bg_weight = 0.4
featuremap_out_channel = 192
test_parameters = dict(conf_threshold=0.3, nms_thres=50, nms_topk=max_lanes)
data_root = "/home/zky/dataset/tusimple"

SNR_SCALE = 1.0
SAMPLE_STEP = 2
HIDDEN_DIM = 256
DIM_DYNAMIC = 64
NUM_DYNAMIC = 2
POOLER_RESOLUTION = 7
num_denoise_query = 2
positive_num = 2
reg_max = 50
aux_weight = 1.0

param_config.SNR_SCALE = SNR_SCALE
param_config.SAMPLE_STEP = SAMPLE_STEP
param_config.HIDDEN_DIM = HIDDEN_DIM
param_config.DIM_DYNAMIC = DIM_DYNAMIC
param_config.NUM_DYNAMIC = NUM_DYNAMIC
param_config.POOLER_RESOLUTION = POOLER_RESOLUTION
param_config.featuremap_out_channel = featuremap_out_channel
param_config.num_denoise_query = num_denoise_query
param_config.positive_num = positive_num
param_config.reg_max = reg_max
param_config.df_loss_weight = df_loss_weight
param_config.angle_loss_weight = angle_loss_weight
param_config.aux_weight = aux_weight

# param_config = get_config("./param.py")

train = get_config("config/common/train.py").train
epochs = 70
batch_size = 20
epoch_per_iter = (3616 // batch_size + 1)
total_iter = epoch_per_iter * epochs
train.max_iter = total_iter
train.checkpointer.period=epoch_per_iter
train.eval_period = epoch_per_iter
train.output_dir = "output_auxhead_reproduce"
# train.init_checkpoint = "output_3000_96.20/model_best.pth"

use_vfl_iter = int(epochs // 2) * epoch_per_iter
# use_vfl_iter = 2 * epoch_per_iter
param_config.use_vfl_iter = use_vfl_iter

model = L(AUX_DiffLane)(
    backbone = L(ResNetWrapper)(
        resnet='resnet34',
        pretrained=True,
        replace_stride_with_dilation=[False, False, False],
        out_conv=False,
    ),
    neck = L(SA_FPN)(
        in_channels=[128, 256, 512],
        out_channels=64,
        num_outs=3,),
    head = L(DiffusionLaneHead)(
        num_priors=800,
        refine_layers=6,
        fc_hidden_dim=64,
        sample_points=36,
        cfg=param_config
    ),
    aux_head = L(AUXCLRHead)(
        num_priors=192,
        refine_layers=3,
        fc_hidden_dim=64,
        sample_points=36,
        cfg=param_config
    ),
    cfg=param_config
)

optimizer = get_config("config/common/optim.py").AdamW
optimizer.lr = 0.3e-3
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
    # L(Normalize)(img_norm=img_norm),
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
    # L(Normalize)(img_norm=img_norm),
    L(ToTensor)(keys=['img'])
]

dataloader = get_config("config/common/tusimple.py").dataloader
dataloader.train.dataset.processes = train_process
dataloader.train.dataset.data_root = data_root
dataloader.train.dataset.cut_height = cut_height
dataloader.train.dataset.cfg = param_config
dataloader.train.total_batch_size = batch_size
dataloader.test.dataset.processes = val_process
dataloader.test.dataset.data_root = data_root
dataloader.test.dataset.cut_height = cut_height
dataloader.test.total_batch_size = batch_size
dataloader.test.dataset.cfg = param_config

# Evaluation config
dataloader.evaluator.output_basedir = "./output_auxhead_reproduce"
dataloader.evaluator.view=False
dataloader.evaluator.test_json_file=os.path.join(data_root,"test_label.json")
