from omegaconf import OmegaConf
from unlanedet.data.tusimple import TuSimple
from unlanedet.config import LazyCall as L
from unlanedet.data.build import build_batch_data_loader
from unlanedet.evaluation import TusimpleEvaluator

ori_img_h = 720
ori_img_w = 1280
img_h = 288
img_w = 800
cut_height=0
test_json_file='./test_label.json'

dataloader = OmegaConf.create()

dataloader.train = L(build_batch_data_loader)(
    dataset = L(TuSimple)(
        data_root = "./tusimple",
        split='trainval',
        cut_height=cut_height,
        processes=None
    ),
    total_batch_size=16,
    num_workers=4,
    shuffle=True,
)

dataloader.test = L(build_batch_data_loader)(
    dataset = L(TuSimple)(
        data_root = "./tusimple",
        split='test',
        cut_height=cut_height,
        processes=None
    ),
    total_batch_size=16,
    num_workers=4,
    drop_last=False,
    shuffle=False,
)

dataloader.evaluator = L(TusimpleEvaluator)(
    ori_img_h=ori_img_h,
    ori_img_w=ori_img_w,
    test_json_file=test_json_file,
    output_basedir="./",
    metric = "acc"    
)
