import numpy as np
from torch.utils.data import Sampler
import torch
# h,w,c
RES_POOL = np.array([
    [1080, 1920, 3],
    [720, 1280, 3],
    [480, 960, 3],
    [478, 960, 3],
    [378, 672, 3], 
    [448, 960, 3],
    [474, 960, 3],
    [368, 640, 3]
])
RES_MAPPING = {
    '0_Road001_Trim003_frames': [1080, 1920], 
    '0_Road001_Trim007_frames': [1080, 1920], 
    '0_Road014_Trim004_frames': [720, 1280], 
    '0_Road014_Trim005_frames': [720, 1280], 
    '0_Road029_Trim002_frames': [480, 960], 
    '0_Road029_Trim003_frames': [480, 960], 
    '0_Road029_Trim004_frames': [480, 960], 
    '0_Road029_Trim005_frames': [480, 960], 
    '0_Road030_Trim001_frames': [478, 960], 
    '0_Road030_Trim002_frames': [478, 960], 
    '0_Road030_Trim003_frames': [478, 960], 
    '0_Road031_Trim001_frames': [480, 960], 
    '0_Road031_Trim003_frames': [480, 960], 
    '0_Road031_Trim004_frames': [480, 960], 
    '0_Road036_Trim004_frames': [480, 960], 
    '0_Road036_Trim005_frames': [480, 960], 
    '125_Road018_Trim005_frames': [1080, 1920], 
    '1269_Road022_Trim002_frames': [1080, 1920], 
    '12_Road014_Trim002_frames': [720, 1280], 
    '12_Road018_Trim004_frames': [1080, 1920], 
    '15_Road001_Trim004_frames': [1080, 1920], 
    '1_Road001_Trim002_frames': [1080, 1920], 
    '1_Road001_Trim005_frames': [1080, 1920], 
    '1_Road012_Trim002_frames': [1080, 1920], 
    '1_Road012_Trim003_frames': [1080, 1920], 
    '1_Road012_Trim004_frames': [1080, 1920], 
    '1_Road013_Trim003_frames': [1080, 1920], 
    '1_Road013_Trim004_frames': [1080, 1920], 
    '1_Road014_Trim001_frames': [720, 1280], 
    '1_Road014_Trim007_frames': [720, 1280], 
    '1_Road015_Trim008_frames': [720, 1280], 
    '1_Road017_Trim002_frames': [1080, 1920], 
    '1_Road017_Trim010_frames': [1080, 1920], 
    '1_Road018_Trim002_frames': [1080, 1920], 
    '1_Road018_Trim006_frames': [1080, 1920], 
    '1_Road018_Trim009_frames': [1080, 1920], 
    '1_Road018_Trim016_frames': [1080, 1920], 
    '1_Road031_Trim005_frames': [480, 960], 
    '1_Road034_Trim003_frames': [478, 960], 
    '25_Road011_Trim005_frames': [1080, 1920], 
    '25_Road015_Trim003_frames': [720, 1280], 
    '25_Road026_Trim004_frames': [1080, 1920], 
    '27_Road006_Trim001_frames': [378, 672], 
    '2_Road001_Trim009_frames': [1080, 1920], 
    '2_Road009_Trim002_frames': [1080, 1920], 
    '2_Road010_Trim001_frames': [1080, 1920], 
    '2_Road010_Trim003_frames': [1080, 1920], 
    '2_Road011_Trim003_frames': [1080, 1920], 
    '2_Road011_Trim004_frames': [1080, 1920], 
    '2_Road012_Trim001_frames': [1080, 1920], 
    '2_Road013_Trim002_frames': [1080, 1920], 
    '2_Road014_Trim003_frames': [720, 1280], 
    '2_Road015_Trim001_frames': [720, 1280], 
    '2_Road015_Trim002_frames': [720, 1280], 
    '2_Road017_Trim001_frames': [1080, 1920], 
    '2_Road018_Trim010_frames': [1080, 1920], 
    '2_Road026_Trim003_frames': [1080, 1920], 
    '3_Road017_Trim007_frames': [1080, 1920], 
    '3_Road017_Trim008_frames': [1080, 1920], 
    '49_Road028_Trim003_frames': [1080, 1920], 
    '4_Road011_Trim001_frames': [1080, 1920], 
    '4_Road011_Trim002_frames': [1080, 1920], 
    '4_Road017_Trim006_frames': [1080, 1920], 
    '4_Road027_Trim006_frames': [1080, 1920], 
    '4_Road027_Trim011_frames': [1080, 1920], 
    '4_Road027_Trim013_frames': [1080, 1920], 
    '4_Road027_Trim015_frames': [1080, 1920], 
    '4_Road028_Trim012_frames': [1080, 1920], 
    '4_Road028_Trim014_frames': [1080, 1920], 
    '5_Road001_Trim001_frames': [1080, 1920], 
    '5_Road017_Trim003_frames': [1080, 1920], 
    '6_Road022_Trim001_frames': [1080, 1920], 
    '78_Road002_Trim001_frames': [368, 640], 
    '7_Road003_Trim001_frames': [448, 960], 
    '8_Road033_Trim001_frames': [474, 960], 
    '8_Road033_Trim002_frames': [474, 960], 
    '8_Road033_Trim003_frames': [474, 960], 
    '8_Road033_Trim004_frames': [474, 960], 
    '9_Road026_Trim002_frames': [1080, 1920], 
    '9_Road028_Trim001_frames': [1080, 1920],
    '0_Road015_Trim008_frames': [720, 1280], 
    '0_Road029_Trim001_frames': [480, 960], 
    '125_Road018_Trim007_frames': [1080, 1920], 
    '1269_Road023_Trim003_frames': [1080, 1920], 
    '12_Road017_Trim005_frames': [1080, 1920], 
    '12_Road018_Trim003_frames': [1080, 1920], 
    '15_Road018_Trim008_frames': [1080, 1920], 
    '1_Road001_Trim006_frames': [1080, 1920], 
    '1_Road010_Trim002_frames': [1080, 1920], 
    '25_Road015_Trim006_frames': [720, 1280], 
    '2_Road017_Trim004_frames': [1080, 1920], 
    '2_Road036_Trim003_frames': [480, 960], 
    '3_Road017_Trim009_frames': [1080, 1920], 
    '4_Road026_Trim001_frames': [1080, 1920], 
    '4_Road027_Trim005_frames': [1080, 1920], 
    '5_Road001_Trim008_frames': [1080, 1920], 
    '6_Road024_Trim001_frames': [1080, 1920], 
    '7_Road005_Trim001_frames': [378, 672], 
    '8_Road033_Trim005_frames': [474, 960], 
    '9_Road028_Trim005_frames': [1080, 1920]
}

def relocate2mid(proposals:np,sps:np,cfg):
    # 把sps之下的值全部变成负数，不改变proposal中代表sp的值
    # proposals[bs *n, 78]
    # sps[bs* n,2] x,y 
    assert len(proposals) == len(sps)
    x_idx=np.linspace(0,cfg.num_points-1,num=cfg.num_points,dtype=np.int)
    anchor_ys = np.linspace(1, 0, num=cfg.num_points).reshape(1,-1)
    new_proposals = proposals.copy()
    reg_x = proposals[:,6:]
    reg_y = anchor_ys.repeat(len(proposals),axis=0)
    
    dist = ((sps[:,0:1].repeat(cfg.num_points,axis=1)-reg_x)**2 + (sps[:,1:].repeat(cfg.num_points,axis=1)-reg_y)**2).reshape(-1,cfg.num_points)
    # dist [bs*n,72]
    min_idx = np.argmin(dist,axis=1).reshape(-1,1)
    min_idx = min_idx.repeat(cfg.num_points,axis=1)
    reg = new_proposals[:,6:]
    reg[x_idx<min_idx]=-10000
    return new_proposals

class CustomSamper(Sampler):
    def __init__(self, data_source) -> None:
        super().__init__(data_source)
        self.data = data_source
    def __iter__(self):
        indices = []
        self.data.folder_all_list = torch.tensor(self.data.folder_all_list)
        for sub_folder in range(len(self.data.sub_folder_name)):
            index = torch.where(self.data.folder_all_list == sub_folder)[0]
            indices.append(index)
        indices = torch.cat(indices,dim=0)
        return iter(indices)

    def __len__(self):
        return len(self.data)

class CustomBatchSampler:
    def __init__(self, sampler, batch_size, drop_last):
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        i = 0
        sampler_list = list(self.sampler)
        for idx in sampler_list:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []

            if (
                i < len(sampler_list) - 1
                and self.sampler.data.folder_all_list[idx]
                != self.sampler.data.folder_all_list[sampler_list[i + 1]]
            ):
                if len(batch) > 0 and not self.drop_last:
                    yield batch
                    batch = []
                else:
                    batch = []
            i += 1
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size
