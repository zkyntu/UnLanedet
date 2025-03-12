from omegaconf import OmegaConf
from unlanedet.config import LazyCall as L
from unlanedet.data.build import build_batch_data_loader,build_batchvil_test_data_loader
from unlanedet.data.vil import VIL100
from unlanedet.evaluation.evaluator import VILEvaluator

# basic setting
ori_img_w=1280
ori_img_h=720
img_w = 800
img_h = 320
# inference setting
max_lanes = 6
cut_height=0

dataloader = OmegaConf.create()

dataloader.train = L(build_batch_data_loader)(
    dataset = L(VIL100)(
        data_root = "./vil",
        split='train',
        cut_height=cut_height,
        processes=None
    ),
    total_batch_size=16,
    num_workers=4,
    shuffle=True,
)

dataloader.test = L(build_batchvil_test_data_loader)(
    dataset = L(VIL100)(
        data_root = "./vil",
        split='test',
        cut_height=cut_height,
        processes=None
    ),
    total_batch_size=16,
    num_workers=4,
    drop_last=False,
    shuffle=False,
)

dataloader.evaluator = L(VILEvaluator)(
    output_basedir="",
    data_root = "",
    split='test',
    metric = "F1"    
)
