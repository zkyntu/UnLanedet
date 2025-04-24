# modified from https://github.com/zkyseu/PPlanedet/blob/v6/pplanedet/model/necks/csprepbifpn.py

import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

def make_divisible(x, divisor):
    return math.ceil(x / divisor) * divisor

def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class SiLU(nn.Module):
    def __init__(self):
        super(SiLU, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)

class Transpose(nn.Module):
    '''Normal Transpose, default for upsampling'''

    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2):
        super().__init__()
        self.upsample_transpose = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=True)

    def forward(self, x):
        return self.upsample_transpose(x)
    
class SimConv(nn.Module):
    """Simplified Conv BN ReLU"""

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 groups=1,
                 bias=False):
        super(SimConv, self).__init__()
        padding = kernel_size // 2

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias = bias)

        self.bn = nn.BatchNorm2d(out_channels,)
        self.act = nn.ReLU()

        self._init_conv(self.conv)
    
    def _init_conv(self,conv:nn.Conv2d):
        bound = 1 / np.sqrt(np.prod(conv.weight.shape[1:]))
        nn.init.uniform_(conv.weight,-bound, bound)
        if conv.bias is not None:
            nn.init.uniform_(conv.bias, -bound, bound)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class BaseConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 ksize,
                 stride,
                 groups=1,
                 bias=False,):
        super(BaseConv, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=(ksize - 1) // 2,
            groups=groups,
            bias=bias)

        self.bn = nn.BatchNorm2d(
            out_channels,)
        self.act = nn.SiLU()
        self._init_conv(self.conv)

    def _init_conv(self,conv:nn.Conv2d):
        bound = 1 / np.sqrt(np.prod(conv.weight.shape[1:]))
        nn.init.uniform_(conv.weight,-bound, bound)
        if conv.bias is not None:
            nn.init.uniform_(conv.bias, -bound, bound)

    def forward(self, x):
        x = self.bn(self.conv(x))
        if self.training:
            y = self.act(x)
        else:
            if isinstance(self.act, nn.SiLU):
                self.act = SiLU()
            y = self.act(x)
        return y

class BiFusion(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.cv1 = SimConv(in_channels[0], out_channels, 1, 1)
        self.cv2 = SimConv(in_channels[1], out_channels, 1, 1)
        self.cv3 = SimConv(out_channels * 3, out_channels, 1, 1)

        self.upsample = Transpose(
            in_channels=out_channels, out_channels=out_channels)
        self.downsample = SimConv(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2)

    def forward(self, x):
        x0 = self.upsample(x[0])
        x1 = self.cv1(x[1])
        x2 = self.downsample(self.cv2(x[2]))
        return self.cv3(torch.cat([x0, x1, x2], 1))
    
class ConvBNReLUBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 groups=1,
                 bias=True):
        super().__init__()
        self.base_block = SimConv(in_channels, out_channels, kernel_size,
                                  stride, groups, bias)

    def forward(self, x):
        return self.base_block(x)


class RepConv(nn.Module):
    '''RepVGGBlock is a basic rep-style block, including training and deploy status
    This code is based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    '''
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1, dilation=1, groups=1, padding_mode='zeros', deploy=False, use_se=False):
        super(RepConv, self).__init__()
        """ Initialization of the class.
        Args:
            in_channels (int): Number of channels in the input image
            out_channels (int): Number of channels produced by the convolution
            kernel_size (int or tuple): Size of the convolving kernel
            stride (int or tuple, optional): Stride of the convolution. Default: 1
            padding (int or tuple, optional): Zero-padding added to both sides of
                the input. Default: 1
            dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
            groups (int, optional): Number of blocked connections from input
                channels to output channels. Default: 1
            padding_mode (string, optional): Default: 'zeros'
            deploy: Whether to be deploy status or training status. Default: False
            use_se: Whether to use se. Default: False
        """
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels

        assert kernel_size == 3
        assert padding == 1

        padding_11 = padding - kernel_size // 2

        self.nonlinearity = nn.ReLU()

        if use_se:
            raise NotImplementedError("se block not supported yet")
        else:
            self.se = nn.Identity()

        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                         padding=padding, dilation=dilation, groups=groups, bias=True, padding_mode=padding_mode)

        else:
            self.rbr_identity = nn.BatchNorm2d(num_features=in_channels) if out_channels == in_channels and stride == 1 else None
            self.rbr_dense = nn.Sequential(* [
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride,  #
                    padding,
                    groups=groups,
                    bias=False),
                nn.BatchNorm2d(out_channels),
            ])
            self.rbr_1x1 = nn.Sequential(* [
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    1,
                    stride,
                    padding_11,  #
                    groups=groups,
                    bias=False),
                nn.BatchNorm2d(out_channels),
            ])

    def forward(self, inputs):
        '''Forward process'''
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.se(self.rbr_reparam(inputs)))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.nonlinearity(self.se(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out))

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _avg_to_3x3_tensor(self, avgp):
        channels = self.in_channels
        groups = self.groups
        kernel_size = avgp.kernel_size
        input_dim = channels // groups
        k = torch.zeros((channels, input_dim, kernel_size, kernel_size))
        k[np.arange(channels), np.tile(np.arange(input_dim), groups), :, :] = 1.0 / kernel_size ** 2
        return k

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch[0].weight
            running_mean = branch[1].running_mean
            running_var = branch[1].running_var
            gamma = branch[1].weight
            beta = branch[1].bias
            eps = branch[1].eps
        elif isinstance(branch, nn.BatchNorm2d):
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
            std = (running_var + eps).sqrt()
            t = (gamma / std).reshape(-1, 1, 1, 1)
            return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        if hasattr(self, 'rbr_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(
            self.rbr_dense[0].in_channels,
            self.rbr_dense[0].out_channels,
            self.rbr_dense[0].kernel_size,
            self.rbr_dense[0].stride,
            padding=self.rbr_dense[0].padding,
            groups=self.rbr_dense[0].groups,
            bias_attr=True)
        
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        self.deploy = True

def get_block(mode):
    if mode == 'repvgg':
        return RepConv
    elif mode == 'conv_silu':
        return ConvBNSiLUBlock
    elif mode == 'conv_relu':
        return ConvBNReLUBlock
    else:
        raise ValueError('Unsupported mode :{}'.format(mode))
    
class ConvBNSiLUBlock(nn.Module):
    # ConvWrapper
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 groups=1,
                 bias=True):
        super().__init__()
        self.base_block = BaseConv(in_channels, out_channels, kernel_size,
                                   stride, groups, bias)

    def forward(self, x):
        return self.base_block(x)

class BaseConv_C3(nn.Module):
    '''Standard convolution in BepC3-Block'''

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super(BaseConv_C3, self).__init__()
        self.conv = nn.Conv2d(
            c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(
            c2,)
        if act == True:
            self.act = nn.ReLU()
        else:
            if isinstance(act, nn.Module):
                self.act = act
            else:
                self.act = nn.Identity()

    def forward(self, x):
        x = self.bn(self.conv(x))
        if self.training:
            y = self.act(x)
        else:
            if isinstance(self.act, nn.SiLU):
                self.act = SiLU()
            y = self.act(x)
        return y

class BottleRep(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 basic_block=RepConv,
                 alpha=True):
        super(BottleRep, self).__init__()
        # basic_block: RepConv or ConvBNSiLUBlock
        self.conv1 = basic_block(in_channels, out_channels)
        self.conv2 = basic_block(out_channels, out_channels)
        if in_channels != out_channels:
            self.shortcut = False
        else:
            self.shortcut = True
        if alpha:
            self.alpha = nn.Parameter(torch.tensor([1],dtype=torch.float32))
        else:
            self.alpha = 1.0

    def forward(self, x):
        outputs = self.conv1(x)
        outputs = self.conv2(outputs)
        return outputs + self.alpha * x if self.shortcut else outputs

class RepLayer_BottleRep(nn.Module):
    """
    RepLayer with RepConvs for M/L, like CSPLayer(C3) in YOLOv5/YOLOX
    named RepBlock in YOLOv6
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_repeats=1,
                 basic_block=RepConv):
        super(RepLayer_BottleRep, self).__init__()
        # in m/l
        self.conv1 = BottleRep(
            in_channels, out_channels, basic_block=basic_block, alpha=True)
        num_repeats = num_repeats // 2
        self.block = nn.Sequential(*(BottleRep(
            out_channels, out_channels, basic_block=basic_block, alpha=True
        ) for _ in range(num_repeats - 1))) if num_repeats > 1 else None

    def forward(self, x):
        x = self.conv1(x)
        if self.block is not None:
            x = self.block(x)
        return x

class BepC3Layer(nn.Module):
    # Beer-mug RepC3 Block, named BepC3 in YOLOv6
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_repeats=1,
                 csp_e=0.5,
                 block=RepConv,
                 act='relu',
                 cfg=None):
        super(BepC3Layer, self).__init__()
        c_ = int(out_channels * csp_e)  # hidden channels
        self.cv1 = BaseConv_C3(in_channels, c_, 1, 1)
        self.cv2 = BaseConv_C3(in_channels, c_, 1, 1)
        self.cv3 = BaseConv_C3(2 * c_, out_channels, 1, 1)
        if block == ConvBNSiLUBlock and act == 'silu':
            self.cv1 = BaseConv_C3(in_channels, c_, 1, 1, act=nn.Silu())
            self.cv2 = BaseConv_C3(in_channels, c_, 1, 1, act=nn.Silu())
            self.cv3 = BaseConv_C3(2 * c_, out_channels, 1, 1, act=nn.Silu())

        self.m = RepLayer_BottleRep(c_, c_, num_repeats, basic_block=block)

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))

class CSPRepBiFPAN(nn.Module):
    """
    CSPRepBiFPAN of YOLOv6 m/l in v3.0
    change lateral_conv + up(Transpose) to BiFusion
    """
    __shared__ = ['depth_mult', 'width_mult', 'act', 'training_mode']

    def __init__(self,
                 depth_mult=1.0,
                 width_mult=1.0,
                 in_channels=[128, 256, 512, 1024],
                 out_channels = 64,
                 training_mode='repvgg',
                 csp_e=0.5,
                 act='relu',
                 cfg= None,
                 num_outs = 3):
        super(CSPRepBiFPAN, self).__init__()
        backbone_ch_list = in_channels
        backbone_num_repeats = [1, 6, 12, 18, 6]

        ch_list = backbone_ch_list + [out_channels for i in range(6)]
        ch_list = [make_divisible(i * width_mult, 8) for i in (ch_list)]

        num_repeats = backbone_num_repeats + [12, 12, 12, 12]
        num_repeats = [(max(round(i * depth_mult), 1) if i > 1 else i)
                       for i in (num_repeats)]

        self.in_channels = in_channels
        self.num_outs = num_outs
        if csp_e == 0.67:
            csp_e = float(2) / 3
        block = get_block(training_mode)
        # RepConv(or RepVGGBlock) in M, but ConvBNSiLUBlock(or ConvWrapper) in L

        # Rep_p4
        self.reduce_layer0 = SimConv(ch_list[3], ch_list[4], 1, 1)
        self.Bifusion0 = BiFusion([ch_list[2], ch_list[1]], ch_list[4])
        self.Rep_p4 = BepC3Layer(
            ch_list[4], ch_list[4], num_repeats[4], csp_e, block=block, act=act)

        # Rep_p3
        self.reduce_layer1 = SimConv(ch_list[4], ch_list[5], 1, 1)
        self.Bifusion1 = BiFusion([ch_list[1], ch_list[0]], ch_list[5])
        self.Rep_p3 = BepC3Layer(
            ch_list[5], ch_list[5], num_repeats[5], csp_e, block=block, act=act)

        # Rep_n3
        self.downsample2 = SimConv(ch_list[5], ch_list[6], 3, 2)
        self.Rep_n3 = BepC3Layer(
            ch_list[5] + ch_list[6],
            ch_list[7],
            num_repeats[6],
            csp_e,
            block=block,
            act=act)

        # Rep_n4
        self.downsample1 = SimConv(ch_list[7], ch_list[8], 3, 2)
        self.Rep_n4 = BepC3Layer(
            ch_list[4] + ch_list[8],
            ch_list[9],
            num_repeats[7],
            csp_e,
            block=block,
            act=act)
        
        #p2 fusion
        self.upsamplep3 = Transpose(ch_list[8],ch_list[8])
        self.reduce_layerp3 = SimConv(ch_list[8],in_channels[0],1,1)
        self.Rep_up_p3 = BepC3Layer(in_channels[0]*2,out_channels,num_repeats[5])

    def forward(self, feats, for_mot=False):
        assert len(feats) == len(self.in_channels)
        [x3, x2, x1, x0] = feats  # p2, p3, p4, p5 

        # top-down FPN
        fpn_out0 = self.reduce_layer0(x0)
        f_concat_layer0 = self.Bifusion0([fpn_out0, x1, x2])
        f_out0 = self.Rep_p4(f_concat_layer0)

        fpn_out1 = self.reduce_layer1(f_out0)
        f_concat_layer1 = self.Bifusion1([fpn_out1, x2, x3])
        pan_out2 = self.Rep_p3(f_concat_layer1)

        # bottom-up PAN
        down_feat1 = self.downsample2(pan_out2)
        p_concat_layer1 = torch.cat([down_feat1, fpn_out1], 1)
        pan_out1 = self.Rep_n3(p_concat_layer1)

        down_feat0 = self.downsample1(pan_out1)
        p_concat_layer2 = torch.cat([down_feat0, fpn_out0], 1)
        pan_out0 = self.Rep_n4(p_concat_layer2)

        #p2 fusion
        up_p3 = self.reduce_layerp3(self.upsamplep3(pan_out2))
        fuse_p2 = torch.cat([x3,up_p3],axis=1)
        pan_outp2 = self.Rep_up_p3(fuse_p2)

        if self.num_outs == 4:
            return [pan_outp2, pan_out2, pan_out1, pan_out0]
        elif self.num_outs == 3:
            return [pan_out2, pan_out1, pan_out0]
        else:
            raise NotImplementedError
