import os
import json
import yaml
from .base_dataset import BaseDataset
from tqdm import tqdm
import numpy as np
import cv2
import os.path as osp
from .transform import DataContainer as DC

class VIL100(BaseDataset):
    def __init__(self, data_root, split, cut_height, processes=None, cfg=None):
        super().__init__(data_root, split, cut_height, processes, cfg)
        dbfile = os.path.join(data_root, 'data', 'db_info.yaml')
        self.imgdir = os.path.join(data_root, 'JPEGImages')
        self.annodir = os.path.join(data_root, 'Annotations')
        self.jsondir = os.path.join(data_root,'Json')
        self.root = data_root
        self.data_infos = []
        self.folder_all_list = []
        self.sub_folder_name = []
        self.max_lane = 0
        with open(dbfile, 'r') as f:
            db = yaml.load(f, Loader=yaml.Loader)['sequences']
            self.info = db
            self.videos = [info['name'] for info in db if info['set'] == split]
        self.load_annotations()

    def get_json_path(self,vid_path):
        json_paths = []
        for root, _, files in os.walk(vid_path):
            for file in files:
                if file.endswith(".json"):
                    json_paths.append(os.path.join(root, file))
        return json_paths

    def load_annotations(self):
        json_paths = []
        self.all_file_name = []
        print("Searching annotation files...")
        for vid in self.videos:
            json_paths.extend(self.get_json_path(os.path.join(self.jsondir,vid)))
        print("Found {} annotations".format(len(json_paths)))
        for json_path in tqdm(json_paths):
            with open(json_path,'r') as jfile:
                data = json.load(jfile)
            self.load_annotation(data)
            self.all_file_name.append(json_path.replace(self.jsondir+'/','')[:-9]+'.lines.txt')
        print('Max lane: {}'.format(self.max_lane))

    def load_annotation(self,data):
        points = []
        lane_id_pool =[]
        image_path = data['info']["image_path"]
        # width,height = cv2.imread(os.path.join(self.imgdir,image_path)).shape[:2]
        mask_path = image_path.split('.')[0] + '.png'
        for lane in data['annotations']['lane']:
            # if lane['lane_id'] not in lane_id_pool:
            points.append(lane['points'])
                # lane_id_pool.append(lane['lane_id'])
        self.data_infos.append(
            dict(
                img_name = os.path.join('JPEGImages',image_path),
                # img_size = [width,height],
                img_path = os.path.join(self.imgdir,image_path),
                mask_path = os.path.join(self.annodir,mask_path),
                lanes = points
            )
        )
        sub_folder = image_path.split('/')[0]
        if sub_folder not in self.sub_folder_name:
            self.sub_folder_name.append(sub_folder)
            # self.mapping.update({sub_folder:[width,height]})
        # using index
        idx = self.sub_folder_name.index(sub_folder)
        self.folder_all_list.append(idx)
        
        
        if len(points) > self.max_lane:
            self.max_lane = len(points)

    def __getitem__(self, idx):
        data_info = self.data_infos[idx]
        if not osp.isfile(data_info['img_path']):
            raise FileNotFoundError('cannot find file: {}'.format(data_info['img_path']))

        img = cv2.imread(data_info['img_path'])

        cut_height = img.shape[0] // 3

        img = img[cut_height:, :, :]
        sample = data_info.copy()
        sample.update({'img': img})

        if self.training:
            label = Image.open(sample['mask_path'])
            label = np.array(label)
            if len(label.shape) > 2:
                label = label[:, :, 0]
            label = label.squeeze()
            label = label[cut_height:, :]
            sample.update({'mask': label})

        sample.update({'cut_height':cut_height})
        sample = self.processes(sample)
        meta = {'full_img_path': data_info['img_path'],
                'img_name': data_info['img_name'],
                'cut_height':cut_height}
        meta = DC(meta, cpu_only=True)
        sample.update({'meta': meta})


        return sample 
