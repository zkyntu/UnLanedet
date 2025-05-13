import os.path as osp
import numpy as np
import cv2
import os
import json
import torchvision
from .base_dataset import BaseDataset
import logging
import random
from .transform import Lanes2ControlPoints
from .transform import DataContainer as DC

SPLIT_FILES = {
    'trainval': ['label_data_0313.json', 'label_data_0601.json', 'label_data_0531.json'],
    'train': ['label_data_0313.json', 'label_data_0601.json'],
    'val': ['label_data_0531.json'],
    'test': ['test_label.json'],
}

class TuSimple(BaseDataset):
    def __init__(self, 
                 data_root, 
                 split, 
                 cut_height=100,
                 processes=None, 
                 cfg=None):
        super().__init__(data_root, split, cut_height, processes, cfg)
        self.anno_files = SPLIT_FILES[split] 
        self.load_annotations()
        self.h_samples = list(range(160, 720, 10))

    def load_annotations(self):
        self.logger.info('Loading TuSimple annotations...')
        self.data_infos = []
        max_lanes = 0
        for anno_file in self.anno_files:
            anno_file = osp.join(self.data_root, anno_file)
            with open(anno_file, 'r') as anno_obj:
                lines = anno_obj.readlines()
            for line in lines:
                data = json.loads(line)
                y_samples = data['h_samples']
                gt_lanes = data['lanes']
                mask_path = data['raw_file'].replace('clips', 'seg_label')[:-3] + 'png'
                lanes = [[(x, y) for (x, y) in zip(lane, y_samples) if x >= 0] for lane in gt_lanes]
                lanes = [lane for lane in lanes if len(lane) > 0]
                max_lanes = max(max_lanes, len(lanes))
                self.data_infos.append({
                    'img_path': osp.join(self.data_root, data['raw_file']),
                    'img_name': data['raw_file'],
                    'mask_path': osp.join(self.data_root, mask_path),
                    'lanes': lanes,
                })

        if self.training:
            random.shuffle(self.data_infos)
        self.max_lanes = max_lanes

class BiazerTusimple(TuSimple):
    def __init__(self, 
                 data_root, 
                 split, 
                 cut_height=100, 
                 processes=None, 
                 cfg=None,
                 cls_agnostic=True):
        super().__init__(data_root, split, cut_height, processes, cfg)
        self.lane_transform = Lanes2ControlPoints(order=cfg.order)
        self.cls_agnostic=cls_agnostic
    def load_annotations(self):
        data_infos = []
        max_lanes = 0
        for anno_file in self.anno_files:
            anno_file = osp.join(self.data_root, anno_file)
            with open(anno_file, 'r') as anno_obj:
                lines = anno_obj.readlines()
            for line in lines:
                data = json.loads(line)
                y_samples = data['h_samples']   # [y0, y1, ...]
                gt_lanes = data['lanes']        # [[x_00, x_01, ...], [x_10, x_11, ...], ...]
                lanes = [[(x, y) for (x, y) in zip(lane, y_samples) if x >= 0] for lane in gt_lanes]
                # lanes: [[(x_00,y0), (x_01,y1), ...], [(x_10,y0), (x_11,y1), ...], ...]
                lanes = [lane for lane in lanes if len(lane) > 0]
                lanes_labels = [0 for lane in lanes]    # 只有一个类别
                max_lanes = max(max_lanes, len(lanes))
                lane_exist = data['lane_exist']
                mask_path = data['raw_file'].replace('clips', 'seg_label')[:-3] + 'png'
                data_infos.append({
                    'img_path': osp.join(self.data_root, data['raw_file']),
                    'img_name': data['raw_file'],
                    'mask_path': osp.join(self.data_root, mask_path),
                    'lanes': lanes,
                    'lanes_labels': np.array(lanes_labels, dtype=np.long),
                    'lane_exist': np.array(lane_exist, dtype=np.long)
                })
        self.data_infos = data_infos
        
        if self.training:
            random.shuffle(self.data_infos)
        self.max_lanes = max_lanes
        
    def __getitem__(self, idx):
        data_info = self.data_infos[idx]
        if not osp.isfile(data_info['img_path']):
            raise FileNotFoundError('cannot find file: {}'.format(data_info['img_path']))

        # sample = self.lane_transform(sample)

        img = cv2.imread(data_info['img_path'])

        img = img[self.cut_height:, :, :]
        sample = data_info.copy()
        sample = self.lane_transform(sample)
        sample.update({'img': img})

        if self.cls_agnostic:
            sample['lanes_labels'] *= 0

        if self.training:
            label = cv2.imread(sample['mask_path'], cv2.IMREAD_UNCHANGED)
            if len(label.shape) > 2:
                label = label[:, :, 0]
            label = label.squeeze()
            label = label[self.cut_height:, :]
            sample.update({'mask': label})

        new_lanes = []
        for lane in sample['lanes']:
            new_lane = []
            for p in lane:
                new_lane.append((p[0], p[1]-self.cut_height))
            new_lanes.append(new_lane)
        sample['lanes'] = new_lanes

        new_lanes2 = []
        for lane in sample['control_points']:
            new_lane = []
            for p in lane:
                new_lane.append((p[0], p[1]-self.cut_height))
            new_lanes2.append(new_lane)
        sample['control_points'] = new_lanes2


        sample = self.processes(sample)
        meta = {'full_img_path': data_info['img_path'],
                'img_name': data_info['img_name']}
        meta = DC(meta, cpu_only=True)
        sample.update({'meta': meta})

        return sample 


        return sample 