from omegaconf import OmegaConf
from unlanedet.data.culane import CULane
from unlanedet.config import LazyCall as L
from unlanedet.data.build import build_batch_data_loader
from unlanedet.evaluation import CULaneEvaluator

ori_img_h = 590 
ori_img_w = 1640 
img_h = 288
img_w = 800
cut_height=0

dataloader = OmegaConf.create()

dataloader.train = L(build_batch_data_loader)(
    dataset = L(CULane)(
        data_root = "./culane",
        split='train',
        cut_height=cut_height,
        processes=None
    ),
    total_batch_size=16,
    num_workers=4,
    shuffle=True,
)

dataloader.test = L(build_batch_data_loader)(
    dataset = L(CULane)(
        data_root = "./culane",
        split='test',
        cut_height=cut_height,
        processes=None
    ),
    total_batch_size=16,
    num_workers=4,
    drop_last=False,
    shuffle=False,
)

dataloader.evaluator = L(CULaneEvaluator)(
    data_root = "./",
    ori_img_h=ori_img_h,
    ori_img_w=ori_img_w,
    output_basedir="./",
    metric = "F1"
)
