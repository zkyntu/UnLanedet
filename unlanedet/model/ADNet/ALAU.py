import torch.nn as nn
from torch.nn.modules.utils import _pair as to_2tuple
from torchvision.ops import DeformConv2d

from ...layers import get_norm,Conv2d

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768, norm_cfg="BN"):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = get_norm(norm_cfg, embed_dim)
        # self.norm = build_norm_layer(norm_cfg, embed_dim)[1]
        self.initialize_layer(self.proj)
    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x
    @staticmethod
    def initialize_layer(layer):
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            nn.init.normal_(layer.weight, mean=0., std=0.001)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)


class ALAU(nn.Module):
    """Adaptive Lane Aware Unit Module.
    Adaptive Lane Aware Unit Module is implemented based on DCN v1.
    It uses anchor shape prediction rather than feature map to
    predict offsets of deform conv layer.
    Args:
        in_channels (int): Number of channels in the input feature map.
        out_channels (int): Number of channels in the output feature map.
        kernel_size (tuple): Deformable conv kernel size.
        deform_groups (int): Deformable conv group size.
        cfg (dict or list[dict], optional): config dict.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=(3,3),
                 deform_groups=4,
                 cfg = None):
        super(ALAU, self).__init__()
        offset_channels = kernel_size[0] * kernel_size[1] * 2
        hidden_layer = (deform_groups * offset_channels) // 2
        self.conv_offset = nn.Sequential(
            PatchEmbed(patch_size=3,stride=2,in_chans=2,embed_dim=hidden_layer),
            PatchEmbed(patch_size=3,stride=2,in_chans=hidden_layer,embed_dim=deform_groups * offset_channels),
        )
        self.conv_adaption = DeformConv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=[(kernel_size[0] - 1) // 2,(kernel_size[1] - 1) // 2],
            groups=deform_groups,)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, shape):
        offset = self.conv_offset(shape.detach())
        x = self.relu(self.conv_adaption(x, offset))
        return x,offset