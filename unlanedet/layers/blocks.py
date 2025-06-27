import torch
import fvcore.nn.weight_init as weight_init
from torch import nn

from .batch_norm import FrozenBatchNorm2d, get_norm
from .wrappers import Conv2d


class CNNBlockBase(nn.Module):
    """
    A CNN block is assumed to have input channels, output channels and a stride.
    The input and output of `forward()` method must be NCHW tensors.
    The method can perform arbitrary computation but must match the given
    channels and stride specification.

    Attribute:
        in_channels (int):
        out_channels (int):
        stride (int):
    """

    def __init__(self, in_channels, out_channels, stride):
        """
        The `__init__` method of any subclass should also contain these arguments.

        Args:
            in_channels (int):
            out_channels (int):
            stride (int):
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

    def freeze(self):
        """
        Make this block not trainable.
        This method sets all parameters to `requires_grad=False`,
        and convert all BatchNorm layers to FrozenBatchNorm

        Returns:
            the block itself
        """
        for p in self.parameters():
            p.requires_grad = False
        FrozenBatchNorm2d.convert_frozen_batchnorm(self)
        return self


class DepthwiseSeparableConv2d(nn.Module):
    """
    A kxk depthwise convolution + a 1x1 convolution.

    In :paper:`xception`, norm & activation are applied on the second conv.
    :paper:`mobilenet` uses norm & activation on both convs.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        padding=1,
        dilation=1,
        stride=1,
        *,
        norm1=None,
        activation1=None,
        norm2=None,
        activation2=None,
    ):
        """
        Args:
            norm1, norm2 (str or callable): normalization for the two conv layers.
            activation1, activation2 (callable(Tensor) -> Tensor): activation
                function for the two conv layers.
        """
        super().__init__()
        self.depthwise = Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            stride = stride,
            bias=not norm1,
            norm=get_norm(norm1, out_channels = in_channels),
            activation=activation1,
        )
        self.pointwise = Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride = stride,
            bias=not norm2,
            norm=get_norm(norm2, out_channels = out_channels),
            activation=activation2,
        )

        # default initialization
        weight_init.c2_msra_fill(self.depthwise)
        weight_init.c2_msra_fill(self.pointwise)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))
    

    
class DilatedBottleneck(nn.Module):
    def __init__(self,
                 in_channels,
                 mid_channels,
                 dilation=1,
                 ):
        super(DilatedBottleneck, self).__init__()
        self.conv1 = Conv2d(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            norm=get_norm("BN",mid_channels),
            activation=nn.ReLU(),
        )

        self.conv2 = Conv2d(
            in_channels=mid_channels,
            out_channels=mid_channels,
            kernel_size=3,
            stride=1,
            dilation=dilation,
            padding=dilation,
            norm=get_norm("BN",mid_channels),
            activation=nn.ReLU(),
        )

        self.conv3 = Conv2d(
            in_channels=mid_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            norm=get_norm("BN",in_channels),
            activation=nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: (B, C, H, W)
        :return:
            out: (B, C, H, W)
        """
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = out + identity

        return out
    
class DilatedBlocks(nn.Module):
    def __init__(self,
                 in_channels=256,
                 mid_channels=64,
                 dilations=[4, 8]
                 ):
        super(DilatedBlocks, self).__init__()
        if isinstance(dilations, int):
            dilations = [dilations, ]

        blocks = []
        for dilation in dilations:
            dilate_bottleneck = DilatedBottleneck(in_channels, mid_channels, dilation)
            blocks.append(dilate_bottleneck)

        self.dilated_blocks = nn.Sequential(*blocks)

    def forward(self, x):
        """
        :param x: (B, C, H, W)
        :return:
            out: (B, C, H, W)
        """
        return self.dilated_blocks(x)