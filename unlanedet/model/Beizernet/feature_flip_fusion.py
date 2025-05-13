import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.ops.deform_conv import DeformConv2d


class DCNV2_Ref(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 group =1,
                 dilation=1):
        super().__init__()
        self.deform_conv = DeformConv2d(in_channels,
                                        out_channels,
                                        kernel_size,
                                        stride,
                                        padding,
                                        groups=group)

        self.conv_offset = nn.Conv2d(
            in_channels * 2,
            group * 3 * kernel_size[0] * kernel_size[1],
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=True)
        
        self.init_weights()
        
    def init_weights(self):
        # super(DCNV2_Ref, self).init_weights()
        if hasattr(self, 'conv_offset'):
            self.conv_offset.weight.data.zero_()
            self.conv_offset.bias.data.zero_()

    def forward(self, x, ref):
        """
        :param x: (B, C, H, W)
        :param ref: (B, C, H, W)
        :return:
        """
#        import pdb;pdb.set_trace()
        concat = torch.cat([x, ref], dim=1)
        out = self.conv_offset(concat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        return self.deform_conv(x, offset, mask)
    
class FeatureFlipFusion(nn.Module):
    def __init__(self, channels):
        super(FeatureFlipFusion, self).__init__()
        self.proj1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(channels)
        )
        self.proj2_conv = DCNV2_Ref(channels, channels, kernel_size=(3, 3), padding=1)
        self.proj2_norm = nn.BatchNorm2d(channels)

    def forward(self, feature):
        """
        :param feature: (B, C, H, W)
        :return:

        """
        flipped = torch.flip(feature, dims=(-1, ))  # 将图像特征图进行翻转, (B, C, H, W)
        feature = self.proj1(feature)      # (B, C, H, W)
        flipped = self.proj2_conv(flipped, feature)     # (B, C, H, W)
        flipped = self.proj2_norm(flipped)      # (B, C, H, W)

        return F.relu(feature + flipped)