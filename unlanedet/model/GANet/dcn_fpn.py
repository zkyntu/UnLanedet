import torch.nn as nn
import torch.nn.functional as F

from ...layers import Conv2d,get_norm
from .lanepoints_conv import LanePointsConv


class DeformFPN(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 out_ids=[],
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 dconv_cfg=None,
                 upsample_cfg=dict(mode='nearest'),
                 dcn_only_cls=False,
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')
                 ):
        super(DeformFPN, self).__init__()
        # assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()
        self.out_ids = out_ids
        self.start_level = start_level
        self.end_level = end_level
        self.dcn_only_cls = dcn_only_cls

        if end_level == -1 or end_level == self.num_ins - 1:
            self.backbone_end_level = self.num_ins
            # assert num_outs >= self.num_ins - start_level
        else:
            # if end_level is not the last level, no extra level is allowed
            self.backbone_end_level = end_level + 1
            assert end_level < self.num_ins
            # assert num_outs == end_level - start_level + 1

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        self.def_convs = nn.ModuleList()

        if dconv_cfg is None:
            feat_channels = 256
            stacked_convs = 3
            num_points = 7
        else:
            feat_channels = dconv_cfg['feat_channels']
            stacked_convs = dconv_cfg['stacked_convs']
            num_points = dconv_cfg['num_points']

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = Conv2d(
                in_channels[i],
                out_channels,
                1,
                norm = get_norm(norm_cfg,out_channels),
                activation=nn.ReLU())
            self.lateral_convs.append(l_conv)

            if i in self.out_ids:
                def_conv = LanePointsConv(
                    in_channels=out_channels,
                    feat_channels=feat_channels,
                    point_feat_channels=out_channels,
                    stacked_convs=stacked_convs,
                    num_points=num_points,
                    gradient_mul=0.1,
                    use_latern=False
                )

                fpn_conv = Conv2d(
                    out_channels,
                    out_channels,
                    3,
                    padding=1,
                    norm = get_norm(norm_cfg,out_channels),
                    activation=nn.ReLU())
            else:
                fpn_conv = None
                def_conv = None

            self.fpn_convs.append(fpn_conv)
            self.def_convs.append(def_conv)

    def forward(self, inputs):
        assert len(inputs) >= len(self.in_channels)

        if len(inputs) > len(self.in_channels):
            for _ in range(len(inputs) - len(self.in_channels)):
                del inputs[0]

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.
            if 'scale_factor' in self.upsample_cfg:
                # fix runtime error of "+=" inplace operation in PyTorch 1.10
                laterals[i - 1] = laterals[i - 1] + F.interpolate(
                    laterals[i], **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] = laterals[i - 1] + F.interpolate(
                    laterals[i], size=prev_shape, **self.upsample_cfg)

            # aux_feat是正常经过采样加和之后的fpn，没有deform的特征.
            if i - 1 == 0:
                aux_feat = laterals[0]

        outs = []
        deform_points = []
        for i in self.out_ids:
            d_feat, points = self.def_convs[i](laterals[i])
            d_feat = self.fpn_convs[i](d_feat)
            outs.append(d_feat)
            deform_points.append(points)

        output = dict(
            features=tuple(outs),
            deform_points=tuple(deform_points)
        )
        if self.dcn_only_cls:
            output.update(aux_feat=aux_feat)

        return output