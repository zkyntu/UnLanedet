from torch import nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url
from ..module.architecture import Detector


class SCNN_AGG(nn.Module):
    def __init__(self, cfg=None):
        super(SCNN, self).__init__()
        self.conv_d = nn.Conv2d(128, 128, (1, 9), padding=(0, 4), bias=False)
        self.conv_u = nn.Conv2d(128, 128, (1, 9), padding=(0, 4), bias=False)
        self.conv_r = nn.Conv2d(128, 128, (9, 1), padding=(4, 0), bias=False)
        self.conv_l = nn.Conv2d(128, 128, (9, 1), padding=(4, 0), bias=False)

    def forward(self, x):
        x = x.clone()
        for i in range(1, x.shape[2]):
            x[..., i:i+1, :] = x[..., i:i+1, :].clone() + F.relu(self.conv_d(x[..., i-1:i, :].clone()))

        for i in range(x.shape[2] - 2, 0, -1):
            x[..., i:i+1, :] = x[..., i:i+1, :].clone() + F.relu(self.conv_u(x[..., i+1:i+2, :].clone()))

        for i in range(1, x.shape[3]):
            x[..., i:i+1] = x[..., i:i+1].clone() + F.relu(self.conv_r(x[..., i-1:i].clone()))

        for i in range(x.shape[3] - 2, 0, -1):
            x[..., i:i+1] = x[..., i:i+1].clone() + F.relu(self.conv_l(x[..., i+1:i+2].clone()))
        return x

class SCNN(Detector):
    def __init__(self, backbone=None, aggregator=None, neck=None, head=None):
        super().__init__(backbone, aggregator, neck, head)
    
    def get_lanes(self, output):
        return super().get_lanes(output)

    def forward(self, batch):
        return super().forward(batch)
