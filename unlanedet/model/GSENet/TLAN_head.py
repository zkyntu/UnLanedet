import torch
from torch import nn
import torch.nn.functional as F
from .tlan_vit import VIT
from .tlan_CNN2 import FeatureExtractor
from .tlan_FlattenMHA import MHA

class TLAN(nn.Module):
    def __init__(self):
        super().__init__()
        self.ViT = VIT(image_size=(10,25),patch_size=(1,5),depth=1,heads=8,dim=2560,mlp_dim=5120)
        self.CNN = FeatureExtractor(512,2)
        self.MHA = MHA(in_channel = 512,head = 8,head_dim = 64,dropout = 0.,out_channel = 2560)
        self.ConV = nn.Conv2d(2048,512,1,1,1)

    def forward(self,x):
        if x.shape[1] == 2048:
            x = self.ConV(x)
        x = self.CNN(x)
        x1 = self.ViT(x)
        x2 = self.MHA(x)

        return x1,x2