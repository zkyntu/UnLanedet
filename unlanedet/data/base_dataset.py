import os.path as osp
import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import torchvision
import logging

from ..model.module.core.lane import Lane
from .transform.transforms import Preprocess
from .transform import DataContainer as DC

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

class BaseDataset(Dataset):
    def __init__(self, data_root, split, cut_height, processes=None,cfg=None):
        self.cfg = cfg
        self.logger = logging.getLogger(__name__)
        self.data_root = data_root
        self.training = 'train' in split 
        self.processes = Preprocess(processes)
        self.cut_height = cut_height

    def view(self, predictions, img_metas,work_dir):
        img_metas = [item for img_meta in img_metas.data for item in img_meta]
        for lanes, img_meta in zip(predictions, img_metas):
            img_name = img_meta['img_name']
            img = cv2.imread(os.path.join(self.data_root, img_name))
            out_file = os.path.join(work_dir, 'visualization',
                                img_name.replace('/', '_'))
            if isinstance(lanes[0],Lane):
                lanes = [lane.to_array(self.cfg) for lane in lanes]
            else:
                lanes = [np.array(lane, dtype=np.float32) for lane in lanes]
            # lanes = [lane.to_array(self.cfg) for lane in lanes]
            imshow_lanes(img, lanes, out_file=out_file)
    
    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, idx):
        data_info = self.data_infos[idx]
        if not osp.isfile(data_info['img_path']):
            raise FileNotFoundError('cannot find file: {}'.format(data_info['img_path']))

        img = cv2.imread(data_info['img_path'])

        img = img[self.cut_height:, :, :]
        sample = data_info.copy()
        sample.update({'img': img})

        if self.training:
            label = cv2.imread(sample['mask_path'], cv2.IMREAD_UNCHANGED)
            if len(label.shape) > 2:
                label = label[:, :, 0]
            label = label.squeeze()
            label = label[self.cut_height:, :]
            sample.update({'mask': label})

        sample = self.processes(sample)
        meta = {'full_img_path': data_info['img_path'],
                'img_name': data_info['img_name']}
        meta = DC(meta, cpu_only=True)
        sample.update({'meta': meta})


        return sample 
