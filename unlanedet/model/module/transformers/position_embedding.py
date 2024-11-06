import math
import torch
import torch.nn as nn
from ....layers import Conv2d,Activation,get_norm
import torch.nn.functional as F

class PositionEmbeddingSine(nn.Module):
    """Sinusoidal position embedding used in DETR model.

    Please see `End-to-End Object Detection with Transformers
    <https://arxiv.org/pdf/2005.12872>`_ for more details.

    Args:
        num_pos_feats (int): The feature dimension for each position along
            x-axis or y-axis. The final returned dimension for each position
            is 2 times of the input value.
        temperature (int, optional): The temperature used for scaling
            the position embedding. Default: 10000.
        scale (float, optional): A scale factor that scales the position
            embedding. The scale will be used only when `normalize` is True.
            Default: 2*pi.
        eps (float, optional): A value added to the denominator for numerical
            stability. Default: 1e-6.
        offset (float): An offset added to embed when doing normalization.
        normalize (bool, optional): Whether to normalize the position embedding.
            Default: False.
    """

    def __init__(
        self,
        num_pos_feats: int = 64,
        temperature: int = 10000,
        scale: float = 2 * math.pi,
        eps: float = 1e-6,
        offset: float = 0.0,
        normalize: bool = False,
    ):
        super().__init__()
        if normalize:
            assert isinstance(scale, (float, int)), (
                "when normalize is set,"
                "scale should be provided and in float or int type, "
                f"found {type(scale)}"
            )
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale
        self.eps = eps
        self.offset = offset

    def forward(self, mask: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward function for `PositionEmbeddingSine`.

        Args:
            mask (torch.Tensor): ByteTensor mask. Non-zero values representing
                ignored positions, while zero values means valid positions
                for the input tensor. Shape as `(bs, h, w)`.

        Returns:
            torch.Tensor: Returned position embedding with
            shape `(bs, num_pos_feats * 2, h, w)`
        """
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            y_embed = (y_embed + self.offset) / (y_embed[:, -1:, :] + self.eps) * self.scale
            x_embed = (x_embed + self.offset) / (x_embed[:, :, -1:] + self.eps) * self.scale
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature ** (
            2 * torch.div(dim_t, 2, rounding_mode="floor") / self.num_pos_feats
        )
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t

        # use view as mmdet instead of flatten for dynamically exporting to ONNX
        B, H, W = mask.size()
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).view(
            B, H, W, -1
        )
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).view(
            B, H, W, -1
        )
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class PositionEmbeddingLearned(nn.Module):
    """
    Position embedding with learnable embedding weights.

    Args:
        num_pos_feats (int): The feature dimension for each position along
            x-axis or y-axis. The final returned dimension for each position
            is 2 times of the input value.
        row_num_embed (int, optional): The dictionary size of row embeddings.
            Default: 50.
        col_num_embed (int, optional): The dictionary size of column embeddings.
            Default: 50.
    """

    def __init__(
        self,
        num_pos_feats: int = 256,
        row_num_embed: int = 50,
        col_num_embed: int = 50,
    ):
        super().__init__()
        self.row_embed = nn.Embedding(row_num_embed, num_pos_feats)
        self.col_embed = nn.Embedding(col_num_embed, num_pos_feats)
        self.num_pos_feats = num_pos_feats
        self.row_num_embed = row_num_embed
        self.col_num_embed = col_num_embed

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, mask):
        """Forward function for `PositionEmbeddingLearned`.

        Args:
            mask (torch.Tensor): ByteTensor mask. Non-zero values representing
                ignored positions, while zero values means valid positions
                for the input tensor. Shape as `(bs, h, w)`.

        Returns:
            torch.Tensor: Returned position embedding with
            shape `(bs, num_pos_feats * 2, h, w)`
        """
        h, w = mask.shape[-2:]
        x = torch.arange(w, device=mask.device)
        y = torch.arange(h, device=mask.device)
        x_emb = self.col_embed(x)
        y_emb = self.row_embed(y)
        pos = (
            torch.cat(
                [
                    x_emb.unsqueeze(0).repeat(h, 1, 1),
                    y_emb.unsqueeze(1).repeat(1, w, 1),
                ],
                dim=-1,
            )
            .permute(2, 0, 1)
            .unsqueeze(0)
            .repeat(mask.shape[0], 1, 1, 1)
        )
        return pos


class AttentionLayer(nn.Module):
    """ Position attention module"""

    def __init__(self, in_dim, out_dim, ratio=4, stride=1):
        super(AttentionLayer, self).__init__()
        self.chanel_in = in_dim
        self.pre_conv = Conv2d(
            in_dim,
            out_dim,
            kernel_size=3,
            stride=stride,
            padding=1,
            norm = get_norm(norm="BN",out_channels=out_dim),
#            activation = nn.ReLU()
        )
        self.query_conv = nn.Conv2d(
            in_channels=out_dim, out_channels=out_dim // ratio, kernel_size=1)
        self.key_conv = nn.Conv2d(
            in_channels=out_dim, out_channels=out_dim // ratio, kernel_size=1)
        self.value_conv = nn.Conv2d(
            in_channels=out_dim, out_channels=out_dim, kernel_size=1)
        self.final_conv = Conv2d(
            out_dim,
            out_dim,    
            kernel_size=3,
            padding=1,    
            norm = get_norm(norm="BN",out_channels=out_dim),
#            activation = nn.ReLU(inplace=False)    
        )
        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x, pos=None):
        """
            inputs :
                x : inpput feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        x = self.pre_conv(x)
        m_batchsize, _, height, width = x.size()
        if pos is not None:
            x += pos
        proj_query = F.relu(self.query_conv(x)).view(m_batchsize, -1,
                                             width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)

        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        attention = attention.permute(0, 2, 1)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)
        out = torch.bmm(proj_value, attention)
        out = out.view(m_batchsize, -1, height, width)
        proj_value = proj_value.view(m_batchsize, -1, height, width)
        out_feat = self.gamma * out + x
        out_feat = F.relu(self.final_conv(out_feat))
        return out_feat

class TransConvEncoderModule(nn.Module):
    def __init__(self, in_dim, attn_in_dims, attn_out_dims, strides, ratios, downscale=True, pos_shape=None, cfg=None):
        super(TransConvEncoderModule, self).__init__()
        if downscale:
            stride = 2
        else:
            stride = 1
        # self.first_conv = ConvModule(in_dim, 2*in_dim, kernel_size=3, stride=stride, padding=1)
        # self.final_conv = ConvModule(attn_out_dims[-1], attn_out_dims[-1], kernel_size=3, stride=1, padding=1)
        attn_layers = []
        for dim1, dim2, stride, ratio in zip(attn_in_dims, attn_out_dims, strides, ratios):
            attn_layers.append(AttentionLayer(dim1, dim2, ratio, stride))
        if pos_shape is not None:
            self.attn_layers = nn.ModuleList(attn_layers)
        else:
            self.attn_layers = nn.Sequential(*attn_layers)
        self.pos_shape = pos_shape
        self.pos_embeds = []
        if pos_shape is not None:
            for dim in attn_out_dims:
                pos_embed = build_position_encoding(dim, pos_shape).cuda()
                self.pos_embeds.append(pos_embed)
    
    def forward(self, src):
        if self.pos_shape is None:
            src = self.attn_layers(src)
        else:
            for layer, pos in zip(self.attn_layers, self.pos_embeds):
                src = layer(src, pos.to(src.device))
        return src

def build_position_encoding(hidden_dim, shape):
#    import pdb;pdb.set_trace()
    mask = torch.zeros(list(shape), dtype=torch.bool)
    pos_module = PositionEmbeddingSine(hidden_dim // 2)
    pos_embs = pos_module(mask)
    return pos_embs

def get_sine_pos_embed(
    pos_tensor: torch.Tensor,
    num_pos_feats: int = 128,
    temperature: int = 10000,
    exchange_xy: bool = True,
) -> torch.Tensor:
    """generate sine position embedding from a position tensor

    Args:
        pos_tensor (torch.Tensor): Shape as `(None, n)`.
        num_pos_feats (int): projected shape for each float in the tensor. Default: 128
        temperature (int): The temperature used for scaling
            the position embedding. Default: 10000.
        exchange_xy (bool, optional): exchange pos x and pos y. \
            For example, input tensor is `[x, y]`, the results will  # noqa 
            be `[pos(y), pos(x)]`. Defaults: True.

    Returns:
        torch.Tensor: Returned position embedding  # noqa 
        with shape `(None, n * num_pos_feats)`.
    """
    scale = 2 * math.pi
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos_tensor.device)
    dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / num_pos_feats)

    def sine_func(x: torch.Tensor):
        sin_x = x * scale / dim_t
        sin_x = torch.stack((sin_x[:, :, 0::2].sin(), sin_x[:, :, 1::2].cos()), dim=3).flatten(2)
        return sin_x

    pos_res = [sine_func(x) for x in pos_tensor.split([1] * pos_tensor.shape[-1], dim=-1)]
    if exchange_xy:
        pos_res[0], pos_res[1] = pos_res[1], pos_res[0]
    pos_res = torch.cat(pos_res, dim=2)
    return pos_res