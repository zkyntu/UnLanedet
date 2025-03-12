import os
import argparse
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import time
import torch
from unlanedet.checkpoint import Checkpointer
from unlanedet.config import LazyConfig, instantiate
from unlanedet.engine import (
    default_setup,
    default_writers,
)
from unlanedet.engine.defaults import create_ddp_model

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='The path of config file')
    parser.add_argument('ckpt', help='The path of checkpoint')
    args = parser.parse_args()

    cfg = LazyConfig.load(args.config)
    cfg = LazyConfig.apply_overrides(cfg, [])
    default_setup(cfg, args)

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    device = cfg.train.device
    repeat_time = 2000

    model = instantiate(cfg.model)
    model.to(device)
    model = create_ddp_model(model)
    model.eval()
    Checkpointer(model).load(args.ckpt)

    input = torch.zeros((1,3,320,800),device=device) # batch size is 1

    data = {'img':input}

    for i in range(200):
        out = model(data) 

    torch.cuda.synchronize()
    start = time.perf_counter()

    for i in tqdm(range(repeat_time)):
        out = model(data)

    torch.cuda.synchronize()
    end = time.perf_counter()
    fps = 1/((end-start) / (repeat_time*1))
    print(fps)
