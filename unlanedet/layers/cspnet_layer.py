import torch
import torch.nn as nn
from torch import Tensor

from .blocks import Conv2d,get_norm,DepthwiseSeparableConv2d

class ChannelAttention(nn.Module):
    """Channel attention Module.

    Args:
        channels (int): The input (and output) channels of the attention layer.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Hardsigmoid(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        """Forward function for ChannelAttention."""
        with torch.cuda.amp.autocast(enabled=False):
            out = self.global_avgpool(x)
        out = self.fc(out)
        out = self.act(out)
        return x * out

class CSPNeXtBlock(nn.Module):
    """The basic bottleneck block used in CSPNeXt.

    Args:
        in_channels (int): The input channels of this Module.
        out_channels (int): The output channels of this Module.
        expansion (float): Expand ratio of the hidden channel. Defaults to 0.5.
        add_identity (bool): Whether to add identity to the out. Only works
            when in_channels == out_channels. Defaults to True.
        use_depthwise (bool): Whether to use depthwise separable convolution.
            Defaults to False.
        kernel_size (int): The kernel size of the second convolution layer.
            Defaults to 5.
        conv_cfg (dict): Config dict for convolution layer. Defaults to None,
            which means using conv2d.
        norm (dict): Config dict for normalization layer.
            Defaults to dict(type='BN', momentum=0.03, eps=0.001).
        act_cfg (dict): Config dict for activation layer.
            Defaults to dict(type='SiLU').
        init_cfg (:obj:`ConfigDict` or dict or list[dict] or
            list[:obj:`ConfigDict`], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 expansion: float = 0.5,
                 add_identity: bool = True,
                 use_depthwise: bool = False,
                 kernel_size: int = 5,) -> None:
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        conv = DepthwiseSeparableConv2d if use_depthwise else Conv2d
        self.conv1 = conv(
            in_channels,
            hidden_channels,
            3,
            stride=1,
            padding=1,
            norm=get_norm("BN",out_channels=hidden_channels),
            activation=nn.SiLU())
        self.conv2 = DepthwiseSeparableConv2d(
            hidden_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=kernel_size // 2,
            norm1="BN",
            activation1=nn.SiLU(),
            norm2="BN",
            activation2=nn.SiLU(),)
        self.add_identity = \
            add_identity and in_channels == out_channels

    def forward(self, x: Tensor) -> Tensor:
        """Forward function."""
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)

        if self.add_identity:
            return out + identity
        else:
            return out


class CSPLayer(nn.Module):
    """Cross Stage Partial Layer.

    Args:
        in_channels (int): The input channels of the CSP layer.
        out_channels (int): The output channels of the CSP layer.
        expand_ratio (float): Ratio to adjust the number of channels of the
            hidden layer. Defaults to 0.5.
        num_blocks (int): Number of blocks. Defaults to 1.
        add_identity (bool): Whether to add identity in blocks.
            Defaults to True.
        use_cspnext_block (bool): Whether to use CSPNeXt block.
            Defaults to False.
        use_depthwise (bool): Whether to use depthwise separable convolution in
            blocks. Defaults to False.
        channel_attention (bool): Whether to add channel attention in each
            stage. Defaults to True.
        conv_cfg (dict, optional): Config dict for convolution layer.
            Defaults to None, which means using conv2d.
        norm (dict): Config dict for normalization layer.
            Defaults to dict(type='BN')
        act_cfg (dict): Config dict for activation layer.
            Defaults to dict(type='Swish')
        init_cfg (:obj:`ConfigDict` or dict or list[dict] or
            list[:obj:`ConfigDict`], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 expand_ratio: float = 0.5,
                 num_blocks: int = 1,
                 add_identity: bool = True,
                 use_depthwise: bool = False,
                 use_cspnext_block: bool = False,
                 channel_attention: bool = False,
                 conv_cfg = None,
                 norm = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg = dict(type='Swish'),
                 init_cfg = None) -> None:
        super().__init__()
        block = CSPNeXtBlock
        mid_channels = int(out_channels * expand_ratio)
        self.channel_attention = channel_attention
        self.main_conv = Conv2d(
            in_channels,
            mid_channels,
            1,
            norm=get_norm("BN",mid_channels),
            activation=nn.SiLU())
        self.short_conv = Conv2d(
            in_channels,
            mid_channels,
            1,
            norm=get_norm("BN",mid_channels),
            activation=nn.SiLU())
        self.final_conv = Conv2d(
            2 * mid_channels,
            out_channels,
            1,
            norm=get_norm("BN",out_channels),
            activation=nn.SiLU())

        self.blocks = nn.Sequential(*[
            block(
                mid_channels,
                mid_channels,
                1.0,
                add_identity,
                use_depthwise,) for _ in range(num_blocks)
        ])
        if channel_attention:
            self.attention = ChannelAttention(2 * mid_channels)

    def forward(self, x: Tensor) -> Tensor:
        """Forward function."""
        x_short = self.short_conv(x)

        x_main = self.main_conv(x)
        x_main = self.blocks(x_main)

        x_final = torch.cat((x_main, x_short), dim=1)

        if self.channel_attention:
            x_final = self.attention(x_final)
        return self.final_conv(x_final)