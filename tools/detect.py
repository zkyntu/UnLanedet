import os
import os.path as osp
import torch
import cv2
import glob
import argparse
import numpy as np
import torch.nn.functional as F
from copy import copy
from tqdm import tqdm
from pathlib import Path
from unlanedet.checkpoint import Checkpointer
from unlanedet.config import LazyConfig, instantiate
from unlanedet.engine import (
    default_setup,
    default_writers,
)
from unlanedet.engine.defaults import create_ddp_model
from unlanedet.data.transform import Preprocess
from unlanedet.model.module.core.lane import Lane

def mkdir(path):
    sub_dir = os.path.dirname(path)
    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir)

def imshow_lanes(img, lanes, show=False, out_file=None):
    for lane in lanes:
        for x, y in lane:
            if x <= 0 or y <= 0:
                continue
            x, y = int(x), int(y)
            cv2.circle(img, (x, y), 4, (255, 0, 0), 2)

    if show:
        cv2.imshow('view', img)
        cv2.waitKey(0)

    if out_file:
        if not osp.exists(osp.dirname(out_file)):
            os.makedirs(osp.dirname(out_file))
        cv2.imwrite(out_file, img)


def preprocess(img_path,cfg,processes):
    ori_img = cv2.imread(img_path)
#    import pdb;pdb.set_trace()
    img = ori_img[cfg.param_config.cut_height:, :, :].astype(np.float32)
    vis_img = copy(img)
    data = {'img': img, 'lanes': [],}
    data = processes(data)
    data['img'] = data['img'].unsqueeze(0)
    data.update({'img_path':img_path, 'ori_img':ori_img,'vis_img': vis_img})
    return data

def show(model,data,cfg):
    with torch.no_grad():
        out = model(data)
#    import pdb;pdb.set_trace()
    data['lanes'] = model.get_lanes(out)[0]
    out_file = cfg.savedir
    if out_file:
        out_file = osp.join(out_file, osp.basename(data['img_path']))
    if isinstance(data['lanes'][0],Lane):
        lanes = [lane.to_array(cfg.param_config) for lane in data['lanes']]
    else:
        lanes = [np.array(lane, dtype=np.float32) for lane in lanes]
    lanes = [lane.to_array(cfg.param_config) for lane in data['lanes']]
    imshow_lanes(data['ori_img'], lanes, show=False, out_file=out_file)

def run(data,cfg,model):
#    import pdb;pdb.set_trace()
    transform = Preprocess(instantiate(cfg.dataloader.test.dataset.processes))
    data = preprocess(data,cfg,transform)
    show(model,data,cfg)    

def get_img_paths(path):
    p = str(Path(path).absolute())  # os-agnostic absolute path
    if '*' in p:
        paths = sorted(glob.glob(p, recursive=True))  # glob
    elif os.path.isdir(p):
        paths = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
    elif os.path.isfile(p):
        paths = [p]  # files
    else:
        raise Exception(f'ERROR: {p} does not exist')
    return paths 

def main(args):
    cfg = LazyConfig.load(args.config)
    cfg = LazyConfig.apply_overrides(cfg, [])
    default_setup(cfg, args)

    model = instantiate(cfg.model)
    model.to(cfg.train.device)
    model = create_ddp_model(model)
    model.eval()
    Checkpointer(model).load(args.ckpt)
    cfg.savedir = args.savedir

    paths = get_img_paths(args.img)
    for p in tqdm(paths):
        run(p,cfg,model)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='The path of config file')
    parser.add_argument('ckpt', help='The path of checkpoint')
    parser.add_argument('--img',  help='The path of the img (img file or img_folder), for example: data/*.png')
    parser.add_argument('--show', action='store_true', 
            help='Whether to show the image')
    parser.add_argument('--savedir', type=str, default=None, help='The root of save directory')
    parser.add_argument('--load_from', type=str, default='best.pth', help='The path of model')
    args = parser.parse_args()
    main(args)
