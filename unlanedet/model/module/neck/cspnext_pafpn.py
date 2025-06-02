# this file is borrowed from mmdet.
import math

import torch
import torch.nn as nn

from ....layers import Conv2d,get_norm,DepthwiseSeparableConv2d
from ....layers.cspnet_layer import CSPLayer
from ....layers.weight_init import kaiming_init
# from .csprepbifpn import

class CSPNeXtPAFPN(nn.Module):
    """Path Aggregation Network with CSPNeXt blocks.

    Args:
        in_channels (Sequence[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        num_csp_blocks (int): Number of bottlenecks in CSPLayer.
            Defaults to 3.
        use_depthwise (bool): Whether to use depthwise separable convolution in
            blocks. Defaults to False.
        expand_ratio (float): Ratio to adjust the number of channels of the
            hidden layer. Default: 0.5
        upsample_cfg (dict): Config dict for interpolate layer.
            Default: `dict(scale_factor=2, mode='nearest')`
        conv_cfg (dict, optional): Config dict for convolution layer.
            Default: None, which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN')
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='Swish')
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    """

    def __init__(
        self,
        in_channels,
        out_channels: int,
        num_csp_blocks: int = 3,
        use_depthwise: bool = False,
        expand_ratio: float = 0.5,
        upsample_cfg = dict(scale_factor=2, mode='nearest'),
        conv_cfg: bool = None,
        init_cfg = dict(
            a=math.sqrt(5),
            distribution='uniform',
            mode='fan_in',
            nonlinearity='leaky_relu')
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        conv = DepthwiseSeparableConv2d if use_depthwise else Conv2d

        # build top-down blocks
        self.upsample = nn.Upsample(**upsample_cfg)
        self.reduce_layers = nn.ModuleList()
        self.top_down_blocks = nn.ModuleList()
        for idx in range(len(in_channels) - 1, 0, -1):
            self.reduce_layers.append(
                Conv2d(
                    in_channels[idx],
                    in_channels[idx - 1],
                    1,
                    norm=get_norm("BN",in_channels[idx - 1]),
                    activation=nn.SiLU()))
            self.top_down_blocks.append(
                CSPLayer(
                    in_channels[idx - 1] * 2,
                    in_channels[idx - 1],
                    num_blocks=num_csp_blocks,
                    add_identity=False,
                    use_depthwise=use_depthwise,
                    use_cspnext_block=True,
                    expand_ratio=expand_ratio,
                    ))

        # build bottom-up blocks
        self.downsamples = nn.ModuleList()
        self.bottom_up_blocks = nn.ModuleList()
        for idx in range(len(in_channels) - 1):
            self.downsamples.append(
                conv(
                    in_channels[idx],
                    in_channels[idx],
                    3,
                    stride=2,
                    padding=1,))
            self.bottom_up_blocks.append(
                CSPLayer(
                    in_channels[idx] * 2,
                    in_channels[idx + 1],
                    num_blocks=num_csp_blocks,
                    add_identity=False,
                    use_depthwise=use_depthwise,
                    use_cspnext_block=True,
                    expand_ratio=expand_ratio,
                    ))

        self.out_convs = nn.ModuleList()
        for i in range(len(in_channels)):
            self.out_convs.append(
                conv(
                    in_channels[i],
                    out_channels,
                    3,
                    padding=1,
                    ))
        self.init_cfg = init_cfg
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m,nn.Conv2d):
            kaiming_init(m,**self.init_cfg)

    def forward(self, inputs):
        """
        Args:
            inputs (tuple[Tensor]): input features.

        Returns:
            tuple[Tensor]: YOLOXPAFPN features.
        """
        assert len(inputs) >= len(self.in_channels)

        if len(inputs) > len(self.in_channels):
            for _ in range(len(inputs) - len(self.in_channels)):
                del inputs[0]

        # top-down path
        inner_outs = [inputs[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_heigh = inner_outs[0]
            feat_low = inputs[idx - 1]
            feat_heigh = self.reduce_layers[len(self.in_channels) - 1 - idx](
                feat_heigh)
            inner_outs[0] = feat_heigh

            upsample_feat = self.upsample(feat_heigh)

            inner_out = self.top_down_blocks[len(self.in_channels) - 1 - idx](
                torch.cat([upsample_feat, feat_low], 1))
            inner_outs.insert(0, inner_out)

        # bottom-up path
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_height = inner_outs[idx + 1]
            downsample_feat = self.downsamples[idx](feat_low)
            out = self.bottom_up_blocks[idx](
                torch.cat([downsample_feat, feat_height], 1))
            outs.append(out)

        # out convs
        for idx, conv in enumerate(self.out_convs):
            outs[idx] = conv(outs[idx])

        return list(outs)