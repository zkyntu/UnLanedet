import torch
from torch import nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url

class PlainDecoder(nn.Module):
    def __init__(self, cfg):
        super(PlainDecoder, self).__init__()
        self.cfg = cfg

        self.dropout = nn.Dropout2d(0.1)
        self.conv8 = nn.Conv2d(cfg.featuremap_out_channel, cfg.num_classes, 1)

    def forward(self, x):

        x = self.dropout(x)
        x = self.conv8(x)
        try:
            x = F.interpolate(x, size=[self.cfg.img_height,  self.cfg.img_width],
                               mode='bilinear', align_corners=False)
        except:
            x = F.interpolate(x, size=[self.cfg.img_h,  self.cfg.img_w],
                               mode='bilinear', align_corners=False)
        

        output = {'seg': x}

        return output 