

import torch
import torch.nn as nn
import torch.nn.init as init
from models.FuzzyLayer_Attention import FuzzyLayer_OR
from torch.cuda.amp import autocast
from models.Fuzzy_module.se_module import SELayer_3d
# from monai.networks.blocks.transformerblock import TransformerBlock
from timm.models.vision_transformer import Block as TransformerBlock
from models.Fuzzy_module.ppm3d import PPM3D
import torch.nn.functional as F
from models.Fuzzy_module.distributed_utils import reduce_value, is_main_process
from models.Fuzzy_module.hog_layer import HOGLayerC
import numpy as np
from scipy import ndimage
import skimage.measure as measure
import pdb, random, math
from copy import deepcopy

def large_connected_domain(label, phase='test1', z_ratio=10):
    # if phase == 'val':
    #     z_ratio = 15
    cd, num = measure.label(label, return_num=True, connectivity=1)
    if num == 0 or phase == 'val':
    # if num == 0:
        return ndimage.binary_fill_holes(label).astype(np.uint8)
    else:
        volume = np.zeros([num])
        for k in range(num):
            volume[k] = ((cd == (k + 1)).astype(np.uint8)).sum()
        volume_sort = np.argsort(volume)
        flag = True
        idex = -1
        while flag:
            label = (cd == (volume_sort[idex] + 1)).astype(np.uint8)
            label_voi = np.where(label > 0)
            z_min, z_max = min(label_voi[0]), max(label_voi[0])
            y_min, y_max = min(label_voi[1]), max(label_voi[1])
            x_min, x_max = min(label_voi[2]), max(label_voi[2])
            z, y, x = z_max-z_min, y_max-y_min,x_max-x_min
            if float(z)/y > z_ratio or float(z)/x > z_ratio:
                print("check this prediction!!!")
                idex += 1
            else:
                flag = False
        label = ndimage.binary_fill_holes(label)
        label = label.astype(np.uint8)
    return label

class ConvDropoutNormNonlin(nn.Module):
    """
    fixes a bug in ConvDropoutNormNonlin where lrelu was used regardless of nonlin. Bad.
    """
    def __init__(self, input_channels, output_channels,
                 conv_op=nn.Conv2d, conv_kwargs=None,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None, se=False):
        super(ConvDropoutNormNonlin, self).__init__()
        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}
        if conv_kwargs is None:
            conv_kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1, 'bias': True}

        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.conv_kwargs = conv_kwargs
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.se = se

        self.conv = self.conv_op(input_channels, output_channels, **self.conv_kwargs)
        if self.se:
            self.se_op = SELayer_3d(output_channels)
        if self.dropout_op is not None and self.dropout_op_kwargs['p'] is not None and self.dropout_op_kwargs[
            'p'] > 0:
            self.dropout = self.dropout_op(**self.dropout_op_kwargs)
        else:
            self.dropout = None
        self.instnorm = self.norm_op(output_channels, **self.norm_op_kwargs)
        self.lrelu = self.nonlin(**self.nonlin_kwargs)

    def forward(self, x):
        x = self.conv(x)
        if self.se:
            x = self.se_op(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return self.lrelu(self.instnorm(x))

class ConvNormNonlinDropout(ConvDropoutNormNonlin):
    def forward(self, x):
        x = self.conv(x)
        if self.se:
            x = self.se_op(x)
        x = self.lrelu(self.instnorm(x))
        if self.dropout is not None:
            x = self.dropout(x)
        return x

class Adaptive_Layer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=[1,1,1], stride=[2,2,2], scale_factor=[0], use_act=False):
        super(Adaptive_Layer, self).__init__()
        self.scale_factor = scale_factor
        self.use_act = use_act
        padding = 1 if kernel_size == 3 else 0
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.norm1 = nn.InstanceNorm3d(out_channels)
        if self.use_act:
            self.act1 = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        scale_factor = [s != 1.0 and s != 0.0 for s in self.scale_factor]
        if any(scale_factor):
            out = F.interpolate(out, scale_factor=self.scale_factor, mode='trilinear', align_corners=True)
        out = self.norm1(out)
        if self.use_act:
            out = self.act1(out)
        return out

class Encoder3D(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, strides=(2,2,2), kernel_size=3, args=None):
        super(Encoder3D, self).__init__()
        self.dropout = args.dropout
        self.se = args.se
        self.sc = args.sc
        self.pw = args.pw
        self.dilation = args.dilation if kernel_size <= 3 else 1
        padding = kernel_size // 2
        if self.dilation > 1:
            self.conv1 = nn.Conv3d(in_channels, middle_channels, kernel_size=kernel_size, stride=strides, padding=padding, dilation=self.dilation)
        else:
            self.conv1 = nn.Conv3d(in_channels, middle_channels, kernel_size=kernel_size, stride=strides, padding=padding)
        self.norm1 = nn.InstanceNorm3d(middle_channels)
        self.act1 = nn.LeakyReLU(inplace=True)
        if self.pw:
            self.conv_pw = ConvDropoutNormNonlin(middle_channels, middle_channels, nn.Conv3d, 
                {'kernel_size': 1, 'stride': 1, 'padding': 0, 'dilation': 1, 'bias': True}, nn.InstanceNorm3d, None, 
                nn.Dropout3d, {'p': self.dropout, 'inplace': True}, se=self.se)
        if self.dilation > 1:
            self.conv2 = nn.Conv3d(middle_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=self.dilation)
        else:
            self.conv2 = nn.Conv3d(middle_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.norm2 = nn.InstanceNorm3d(out_channels)
        self.act2 = nn.LeakyReLU(inplace=True)
        if self.se:
            self.se_op0 = SELayer_3d(middle_channels)
            self.se_op1 = SELayer_3d(out_channels)
        if self.sc:
            # self.shortcut = Adaptive_Layer(in_channels, out_channels, stride=strides)
            sc_stride = 1
            scale_factor = [1.0/ii for ii in strides]
            self.shortcut = Adaptive_Layer(in_channels, out_channels, stride=sc_stride, scale_factor=scale_factor)

        if self.dropout:
            assert 0 <= self.dropout and self.dropout <= 1, 'self.dropout must be between 0 and 1'
            self.drop0 = nn.Dropout3d(p=self.dropout)
            self.drop1 = nn.Dropout3d(p=self.dropout)

    def forward(self, x):
        e0 = self.conv1(x)
        e0 = self.norm1(e0)
        e0 = self.act1(e0)
        if self.dropout:
            e0 = self.drop0(e0)
        if self.pw:
            e0 = self.conv_pw(e0)
        if self.se:
            e0 = self.se_op0(e0)
        e1 = self.conv2(e0)
        e1 = self.norm2(e1)
        if self.sc:
            e1 = e1 + self.shortcut(x)
        e1 = self.act2(e1)
        if self.dropout:
            e1 = self.drop1(e1)
        if self.se:
            e1 = self.se_op1(e1)
        return e0, e1


class Center3D(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, deconv_channels, Up_method, 
                 strides=(2,2,2), args=None):
        super(Center3D, self).__init__()
        self.args = args
        self.dropout = args.dropout
        self.se = args.se
        self.pw = args.pw
        self.mhsa_n = args.mhsa_n
        self.uper = args.uper
        self.dilation = args.dilation
        self.conv1 = nn.Conv3d(in_channels, middle_channels, kernel_size=3, padding=1, stride=strides)
        self.norm1 = nn.InstanceNorm3d(middle_channels)
        self.act1 = nn.LeakyReLU(inplace=True)
        if self.pw:
            self.conv_pw = ConvDropoutNormNonlin(middle_channels, middle_channels, nn.Conv3d, 
                {'kernel_size': 1, 'stride': 1, 'padding': 0, 'dilation': 1, 'bias': True}, nn.InstanceNorm3d, None, 
                nn.Dropout3d, {'p': self.dropout, 'inplace': True}, se=self.se)
        if self.dilation > 1:
            self.conv2 = nn.Conv3d(middle_channels, out_channels, kernel_size=3, padding='same', dilation=self.dilation)
        else:
            self.conv2 = nn.Conv3d(middle_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.InstanceNorm3d(out_channels)
        self.act2 = nn.LeakyReLU(inplace=True)
        if Up_method == 'ConvTrans':
            self.up = nn.ConvTranspose3d(out_channels, deconv_channels, kernel_size=2, stride=strides)
        elif Up_method == 'Upsampling':
            self.up = nn.Upsample(scale_factor=strides, mode='trilinear', align_corners=True)
        if self.se:
            self.se_op0 = SELayer_3d(middle_channels)
            self.se_op1 = SELayer_3d(out_channels)
        if self.mhsa_n:
            self.transformers0 = nn.Sequential(
                # *[TransformerBlock(in_channels, 512, 8, 0.0) for ii in range(self.mhsa_n)]
                *[TransformerBlock(in_channels, 8) for ii in range(self.mhsa_n)]
            )
            self.transformers1 = nn.Sequential(
                # *[TransformerBlock(out_channels, 512, 8, 0.0) for ii in range(self.mhsa_n)]
                *[TransformerBlock(out_channels, 8) for ii in range(self.mhsa_n)]
            )
            self.dw_conv1 = nn.Conv3d(in_channels, in_channels, kernel_size=7, 
                                   padding=3, stride=1, groups=in_channels)
            self.dw_norm1 = nn.InstanceNorm3d(in_channels)
            self.dw_act1 = nn.LeakyReLU(inplace=True)
            self.pw_conv1 = nn.Conv3d(in_channels, in_channels, kernel_size=1, 
                                   padding=0, stride=1, groups=1)
            self.pw_norm1 = nn.InstanceNorm3d(in_channels)
            self.pw_act1 = nn.LeakyReLU(inplace=True)
            self.dw_conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=7, 
                                   padding=3, stride=1, groups=out_channels)
            self.dw_norm2 = nn.InstanceNorm3d(out_channels)
            self.dw_act2 = nn.LeakyReLU(inplace=True)
            self.pw_conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=1, 
                                   padding=0, stride=1, groups=1)
            self.pw_norm2 = nn.InstanceNorm3d(out_channels)
            self.pw_act2 = nn.LeakyReLU(inplace=True)
        
        if self.uper:
            self.pool_scales = [1,3,6]
            # self.in_sizes = [16,12,18]
            self.in_sizes = [max(int(math.ceil(self.args.input_size[0]/float(self.args.cum_pool_scales[3][0]))), 1), 
                             max(int(math.ceil(self.args.input_size[1]/float(self.args.cum_pool_scales[3][1]))), 1), 
                             max(int(math.ceil(self.args.input_size[2]/float(self.args.cum_pool_scales[3][2]))), 1)]
            self.ppm3d = PPM3D(self.pool_scales,
                               in_channels, 
                               out_channels, 
                               self.in_sizes)
            self.conv3 = nn.Conv3d((len(self.pool_scales)+1)*out_channels, out_channels, kernel_size=3, padding=1)
            self.norm3 = nn.InstanceNorm3d(out_channels)
            self.act3 = nn.LeakyReLU(inplace=True)

        if self.dropout:
            assert 0 <= self.dropout and self.dropout <= 1, 'self.dropout must be between 0 and 1'
            self.drop0 = nn.Dropout3d(p=self.dropout)
            self.drop1 = nn.Dropout3d(p=self.dropout)

    def forward(self, x):
        # pdb.set_trace()
        if self.mhsa_n:
            x11, x12 = x.clone(), x.clone()
            B,C,D,H,W = x11.size()
            x11 = x11.permute(0,2,3,4,1).flatten(1,3)
            x11 = self.transformers0(x11)
            x11 = x11.contiguous().view(B,D,H,W,C).permute(0,4,1,2,3)
            x12 = self.dw_act1(self.dw_norm1(self.dw_conv1(x12)))
            x12 = self.pw_act1(self.pw_norm1(self.pw_conv1(x12)))
            x = x11 + x12
        
        c = self.conv1(x)
        c = self.norm1(c)
        c = self.act1(c)
        if self.dropout:
            c = self.drop0(c)
        if self.pw:
            c = self.conv_pw(c)
        if self.se:
            c = self.se_op0(c)
        c = self.conv2(c)
        c = self.norm2(c)
        c = self.act2(c)
        if self.dropout:
            c = self.drop1(c)
        if self.se:
            c = self.se_op1(c)
        # pdb.set_trace()
        if self.mhsa_n:
            c11, c12 = c.clone(), c.clone()
            B,C,D,H,W = c11.size()
            c11 = c11.permute(0,2,3,4,1).flatten(1,3)
            # print('\n1. mhsa, c11.size: {}'.format(c11.size()))
            c11 = self.transformers1(c11)
            # print('2. mhsa, c11.size: {}'.format(c11.size()))
            c11 = c11.contiguous().view(B,D,H,W,C).permute(0,4,1,2,3)
            # print('3. mhsa, c11.size: {}'.format(c11.size()))
            c12 = self.dw_act2(self.dw_norm2(self.dw_conv2(c12)))
            c12 = self.pw_act2(self.pw_norm2(self.pw_conv2(c12)))
            c = c11 + c12
        c = self.up(c)
        # pdb.set_trace()
        if self.uper:
            ppm3d_out = self.ppm3d(x)
            ppm3d_out = torch.cat((c, ppm3d_out), dim=1)
            c = self.conv3(ppm3d_out)
            c = self.norm3(c)
            c = self.act3(c)
        return c


class Decoder3D(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, deconv_channels, Up_method, strides=[2,2,2], args=None):
        super(Decoder3D, self).__init__()
        self.dropout = args.dropout
        self.se = args.se
        self.sc = args.sc
        self.pw = args.pw
        self.dilation = args.dilation
        if self.dilation > 1:
            self.conv1 = nn.Conv3d(in_channels, middle_channels, kernel_size=3, padding='same', dilation=self.dilation)
        else:
            self.conv1 = nn.Conv3d(in_channels, middle_channels, kernel_size=3, padding=1, stride=1)
        self.norm1 = nn.InstanceNorm3d(middle_channels)
        self.act1 = nn.LeakyReLU(inplace=True)
        if self.pw:
            self.conv_pw = ConvDropoutNormNonlin(middle_channels, middle_channels, nn.Conv3d, 
                {'kernel_size': 1, 'stride': 1, 'padding': 0, 'dilation': 1, 'bias': True}, nn.InstanceNorm3d, None, 
                nn.Dropout3d, {'p': self.dropout, 'inplace': True}, se=self.se)
        if self.dilation > 1:
            self.conv2 = nn.Conv3d(middle_channels, out_channels, kernel_size=3, padding='same', dilation=self.dilation)
        else:
            self.conv2 = nn.Conv3d(middle_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.InstanceNorm3d(out_channels)
        self.act2 = nn.LeakyReLU(inplace=True)
        if self.sc:
            self.shortcut = Adaptive_Layer(in_channels, out_channels, stride=1)
        if Up_method == 'ConvTrans':
            self.up = nn.ConvTranspose3d(out_channels, deconv_channels, kernel_size=2, stride=strides)
        elif Up_method == 'Upsampling':
            self.up = nn.Upsample(scale_factor=strides, mode='trilinear', align_corners=True)
        if self.se:
            self.se_op0 = SELayer_3d(middle_channels)
            self.se_op1 = SELayer_3d(out_channels)
        if self.dropout:
            assert 0 <= self.dropout and self.dropout <= 1, 'self.dropout must be between 0 and 1'
            self.drop0 = nn.Dropout3d(p=self.dropout)
            self.drop1 = nn.Dropout3d(p=self.dropout)

    def forward(self, x):
        c = self.conv1(x)
        c = self.norm1(c)
        c = self.act1(c)
        if self.dropout:
            c = self.drop0(c)
        if self.pw:
            c = self.conv_pw(c)
        if self.se:
            c = self.se_op0(c)
        c = self.conv2(c)
        c = self.norm2(c)
        if self.sc:
            # pdb.set_trace()
            c = c + self.shortcut(x)
        c = self.act2(c)
        if self.dropout:
            c = self.drop1(c)
        if self.se:
            c = self.se_op1(c)
        up = self.up(c)
        return c, up


class Output_Layer(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, up_scale, se=False, args=None):
        super(Output_Layer, self).__init__()
        self.se = se
        self.args = args
        self.conv1 = nn.Conv3d(in_channels, middle_channels, kernel_size=3, padding=1)
        self.norm1 = nn.InstanceNorm3d(middle_channels)
        self.act1 = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv3d(middle_channels, out_channels, kernel_size=1)
        self.out_act = nn.Sigmoid()
        self.up_sample = nn.Upsample(scale_factor=up_scale, mode='trilinear', align_corners=True)
        if self.se:
            self.se_op0 = SELayer_3d(middle_channels)
            # self.se_op1 = SELayer_3d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act1(out)
        if self.se:
            out = self.se_op0(out)
        if self.args.ds_up_conv:
            out = self.up_sample(out)
            out_t = out.clone()
            # if self.se:
            #     out = self.se_op1(out)
            out = self.conv2(out)
            out = self.out_act(out)
        else:
            out_t = out.clone()
            out = self.conv2(out)
            # if self.se:
            #     out = self.se_op1(out)
            out = self.out_act(out)
            out = self.up_sample(out)
            out_t = self.up_sample(out_t)
        return out, out_t


class Last_Layer(nn.Module):
    def __init__(self, channels, se=False, sc=False, mmgl=False, args=None):
        super(Last_Layer, self).__init__()
        self.se = se
        self.sc = sc
        self.mmgl = mmgl
        self.args = args
        self.conv1 = nn.Conv3d(channels[0], channels[1], kernel_size=3, padding=1)
        self.norm1 = nn.InstanceNorm3d(channels[1])
        self.act1 = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv3d(channels[1], channels[2], kernel_size=3, padding=1)
        self.norm2 = nn.InstanceNorm3d(channels[2])
        self.act2 = nn.LeakyReLU(inplace=True)
        if self.sc:
            self.shortcut = Adaptive_Layer(channels[0], channels[2], stride=1)
        self.conv3 = nn.Conv3d(channels[2], channels[3], kernel_size=1)
        self.out_act = nn.Sigmoid()
        if self.se:
            self.se_op0 = SELayer_3d(channels[1])
            self.se_op1 = SELayer_3d(channels[2])

    def forward(self, x):
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act1(out)
        if self.se:
            out = self.se_op0(out)
        out = self.conv2(out)
        out = self.norm2(out)
        if self.sc:
            out = out + self.shortcut(x)
        out = self.act2(out)
        if self.se:
            out = self.se_op1(out)
        # mmgl_out = None
        # if self.mmgl:
        mmgl_out = out.clone()
        out = self.conv3(out)
        out = self.out_act(out)
        return [out, mmgl_out]

class SSL_Output_Layer(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, up_scale, se=False, one_conv=False):
        super(SSL_Output_Layer, self).__init__()
        self.se = se
        self.one_conv = one_conv
        if self.one_conv:
            self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.conv1 = nn.Conv3d(in_channels, middle_channels, kernel_size=3, padding=1)
            self.norm1 = nn.InstanceNorm3d(middle_channels)
            self.act1 = nn.LeakyReLU(inplace=True)
            self.conv2 = nn.Conv3d(middle_channels, out_channels, kernel_size=1)
            if self.se:
                self.se_op0 = SELayer_3d(middle_channels)
                # self.se_op1 = SELayer_3d(out_channels)
        self.up_sample = nn.Upsample(scale_factor=up_scale, mode='trilinear', align_corners=True)

    def forward(self, x):
        if self.one_conv:
            out = self.up_sample(x)
            out = self.conv1(out)
        else:
            out = self.conv1(x)
            out = self.norm1(out)
            out = self.act1(out)
            if self.se:
                out = self.se_op0(out)
            out = self.up_sample(out)
            out = self.conv2(out)
            # if self.se:
            #     out = self.se_op1(out)
        return out

class SSL_Last_Layer(nn.Module):
    def __init__(self, channels, se=False, one_conv=False):
        super(SSL_Last_Layer, self).__init__()
        self.se = se
        self.one_conv = one_conv
        if self.one_conv:
            self.conv1 = nn.Conv3d(channels[0], channels[2], kernel_size=3, padding=1)
            self.norm1 = nn.InstanceNorm3d(channels[2])
            self.act1 = nn.LeakyReLU(inplace=True)
            self.conv2 = nn.Conv3d(channels[2], channels[3], kernel_size=1)
            if self.se:
                self.se_op0 = SELayer_3d(channels[2])
        else:
            self.conv1 = nn.Conv3d(channels[0], channels[1], kernel_size=3, padding=1)
            self.norm1 = nn.InstanceNorm3d(channels[1])
            self.act1 = nn.LeakyReLU(inplace=True)
            self.conv2 = nn.Conv3d(channels[1], channels[2], kernel_size=3, padding=1)
            self.norm2 = nn.InstanceNorm3d(channels[2])
            self.act2 = nn.LeakyReLU(inplace=True)
            self.conv3 = nn.Conv3d(channels[2], channels[3], kernel_size=1)
            if self.se:
                self.se_op0 = SELayer_3d(channels[1])
                self.se_op1 = SELayer_3d(channels[2])

    def forward(self, x):
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act1(out)
        if self.se:
            out = self.se_op0(out)
        out = self.conv2(out)
        if not self.one_conv:
            out = self.norm2(out)
            out = self.act2(out)
            if self.se:
                out = self.se_op1(out)
            out = self.conv3(out)
        return out

# must filter_e == filter_mix, may filter_d != filter_mix
# i.e. the channels of encoder features must be equal to filter_mix
class FuzzyAttention_Layer(nn.Module):
    def __init__(self, filter_d, filter_e, filter_mix, fuzzy_num, se=False, sc=False):
        super(FuzzyAttention_Layer, self).__init__()
        self.se = se
        self.sc = sc
        self.conv1 = nn.Conv3d(filter_d, filter_mix, kernel_size=1)
        self.norm1 = nn.InstanceNorm3d(filter_mix)
        self.act1 = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv3d(filter_e, filter_mix, kernel_size=1)
        self.norm2 = nn.InstanceNorm3d(filter_mix)
        self.act2 = nn.LeakyReLU(inplace=True)
        self.relu = nn.LeakyReLU(inplace=True)
        self.fuzzyattention = FuzzyLayer_OR(filter_mix,filter_mix,fuzzy_num)
        if self.se:
            self.se_op0 = SELayer_3d(filter_mix)
            self.se_op1 = SELayer_3d(filter_mix)

    def forward(self, e, d):
        d1 = self.act1(self.norm1(self.conv1(d)))
        e1 = self.act2(self.norm2(self.conv2(e)))
        if self.se:
            d1 = self.se_op0(d1)
            e1 = self.se_op1(e1)
        fusion = self.relu(d1+e1)
        fusion = self.fuzzyattention(fusion)
        out = e*fusion
        return out


class FuzzyAttention_3DUNet(nn.Module):
    def __init__(self, in_channel=1, n_classes=1, Up_method='Upsampling', use_mae=False, use_hog=False, 
                 hog_channels=3, args=None):
        super(FuzzyAttention_3DUNet, self).__init__()
        self.in_channel = in_channel
        self.stages_chs = (32, 64, 128, 256, 512) # stages channels
        self.n_classes = n_classes
        self.use_hog = use_hog
        self.use_mae = use_mae
        self.mo_hog_weights = args.mo_hog_weights
        self.mo_mae_weights = args.mo_mae_weights
        self.args = args
        self.se = args.se
        self.sc = args.sc
        self.mhsa_n = args.mhsa_n
        self.uper = args.uper
        self.mmgl = args.mmgl
        self.vote_weights = args.vote_weights
        self.max_num_features = self.stages_chs[-1]
        args_ec0 = deepcopy(args)
        args_ec0.dropout, args_ec0.se, args_ec0.sc, args_ec0.dilation, args_ec0.pw = 0.0, False, False, 1, False
        self.ec0 = Encoder3D(self.in_channel, 16, self.stages_chs[0], strides=self.args.pool_op_kernel_sizes[0], 
            kernel_size=self.args.stem_kernel_size, args=args_ec0) # 32, w, h
        # self.ec0 = Encoder3D(self.in_channel, self.stages_chs[0], self.stages_chs[0], strides=self.args.pool_op_kernel_sizes[0], 
        #     kernel_size=self.args.stem_kernel_size, args=args_ec0) # 32, w, h
        # self.ec1 = Encoder3D(self.stages_chs[0], self.stages_chs[1], self.stages_chs[1], strides=self.args.pool_op_kernel_sizes[1], args=self.args) # 64, w/2, h/2
        # self.ec2 = Encoder3D(self.stages_chs[1], self.stages_chs[2], self.stages_chs[2], strides=self.args.pool_op_kernel_sizes[2], args=self.args)  # 128, w/4, h/4
        # self.ec3 = Encoder3D(self.stages_chs[2], self.stages_chs[3], self.stages_chs[3], strides=self.args.pool_op_kernel_sizes[3], args=self.args)  # 256, w/8, h/8
        self.ec1 = Encoder3D(self.stages_chs[0], self.stages_chs[1], self.stages_chs[1], strides=self.args.pool_op_kernel_sizes[1], 
            kernel_size=self.args.stem_kernel_size, args=self.args) # 64, w/2, h/2
        self.ec2 = Encoder3D(self.stages_chs[1], self.stages_chs[2], self.stages_chs[2], strides=self.args.pool_op_kernel_sizes[2], 
            kernel_size=self.args.stem_kernel_size, args=self.args)  # 128, w/4, h/4
        self.ec3 = Encoder3D(self.stages_chs[2], self.stages_chs[3], self.stages_chs[3], strides=self.args.pool_op_kernel_sizes[3], 
            kernel_size=self.args.stem_kernel_size, args=self.args)  # 256, w/8, h/8
        self.c = Center3D(self.stages_chs[3], self.max_num_features, self.stages_chs[3], self.stages_chs[3], 
                          Up_method, strides=self.args.pool_op_kernel_sizes[4], args=self.args)  # 256, w/8, h/8
        
        self.att0 = FuzzyAttention_Layer(self.stages_chs[3], self.stages_chs[3], self.stages_chs[3], 4, se=self.se, sc=self.sc)
        self.dc0 = Decoder3D(self.max_num_features, self.stages_chs[2], self.stages_chs[2], self.stages_chs[2], Up_method, 
                             strides=self.args.pool_op_kernel_sizes[3], args=self.args)
        if self.uper:
            self.out0 = Output_Layer(self.stages_chs[2], 32, n_classes, up_scale=self.args.cum_pool_scales[3], se=self.se, args=self.args)
        else:
            self.out0 = Output_Layer(self.stages_chs[2], 64, n_classes, up_scale=self.args.cum_pool_scales[3], se=self.se, args=self.args)

        self.att1 = FuzzyAttention_Layer(self.stages_chs[2], self.stages_chs[2], self.stages_chs[2], 4, se=self.se, sc=self.sc)
        self.dc1 = Decoder3D(self.stages_chs[3], self.stages_chs[1], self.stages_chs[1], self.stages_chs[1], Up_method, 
                             strides=self.args.pool_op_kernel_sizes[2], args=self.args)
        self.out1 = Output_Layer(self.stages_chs[1], self.stages_chs[0], n_classes, up_scale=self.args.cum_pool_scales[2], 
                                 se=self.se, args=self.args)

        self.att2 = FuzzyAttention_Layer(self.stages_chs[1], self.stages_chs[1], self.stages_chs[1], 
                                         4, se=self.se, sc=self.sc)
        self.dc2 = Decoder3D(self.stages_chs[2], self.stages_chs[0], self.stages_chs[0], self.stages_chs[0], Up_method, 
                             strides=self.args.pool_op_kernel_sizes[1], args=self.args)
        if self.uper:
            self.out2 = Output_Layer(self.stages_chs[0], self.stages_chs[0], n_classes, up_scale=self.args.cum_pool_scales[1], 
                se=self.se, args=self.args)
        else:
            self.out2 = Output_Layer(self.stages_chs[0], self.stages_chs[0]//2, n_classes, up_scale=self.args.cum_pool_scales[1], 
                se=self.se, args=self.args)

        self.att3 = FuzzyAttention_Layer(self.stages_chs[0], self.stages_chs[0], self.stages_chs[0], 
                                         4, se=self.se, sc=self.sc)
        # self.out = Last_Layer([64, 32, 32, self.n_classes], se=self.se, args=self.args)
        if self.uper:
            # self.out = Last_Layer([self.stages_chs[0]*5, 32, 32, self.n_classes], se=self.se, sc=self.sc, mmgl=self.mmgl, args=self.args)
            self.out = Last_Layer([self.stages_chs[0]*5, 80, 32, self.n_classes], se=self.se, sc=True, mmgl=self.mmgl, args=self.args)
            # self.out = Last_Layer([self.stages_chs[0]*5, 80, 32, self.n_classes], se=self.se, sc=False, mmgl=self.mmgl, args=self.args)
        else:
            self.out = Last_Layer([self.stages_chs[1], 32, 32, self.n_classes], se=self.se, sc=self.sc, mmgl=self.mmgl, args=self.args)
            # self.out = Last_Layer([self.stages_chs[1], 32, 32, self.n_classes], se=self.se, sc=False, mmgl=self.mmgl, args=self.args)
        
        if self.use_mae:
            self.mae_out0 = SSL_Output_Layer(self.stages_chs[2], self.stages_chs[2]//2, in_channel, 
                                             up_scale=self.args.cum_pool_scales[3], se=self.se, one_conv=self.args.ssl_one_conv)
            self.mae_out1 = SSL_Output_Layer(self.stages_chs[1], self.stages_chs[1]//2, in_channel, 
                                             up_scale=self.args.cum_pool_scales[2], se=self.se, one_conv=self.args.ssl_one_conv)
            self.mae_out2 = SSL_Output_Layer(self.stages_chs[0], self.stages_chs[0]//2, in_channel, 
                                             up_scale=self.args.cum_pool_scales[1], se=self.se, one_conv=self.args.ssl_one_conv)
            self.mae_out = SSL_Last_Layer([self.stages_chs[1], 32, 32, in_channel], se=self.se, one_conv=self.args.ssl_one_conv)
        
        if self.use_hog:
            # self.hog_out0 = SSL_Output_Layer(self.stages_chs[2], self.stages_chs[2]//2, hog_channels * in_channel, 
            #   up_scale=self.args.cum_pool_scales[3], se=self.se, one_conv=self.args.ssl_one_conv) # merging multi-modalities prediction by multiplying in_channel, se=self.se.
            self.hog_out0 = SSL_Output_Layer(self.stages_chs[2], self.stages_chs[2]//2, hog_channels * in_channel, 
                up_scale=self.args.cum_pool_scales[3], one_conv=self.args.ssl_one_conv) # merging multi-modalities prediction by multiplying in_channel, se=self.se.
            self.hog_out1 = SSL_Output_Layer(self.stages_chs[1], self.stages_chs[1]//2, hog_channels * in_channel, 
                up_scale=self.args.cum_pool_scales[2], se=self.se, one_conv=self.args.ssl_one_conv)
            self.hog_out2 = SSL_Output_Layer(self.stages_chs[0], self.stages_chs[0]//2, hog_channels * in_channel, 
                up_scale=self.args.cum_pool_scales[1], se=self.se, one_conv=self.args.ssl_one_conv)
            self.hog_out = SSL_Last_Layer([self.stages_chs[1], 32, 32, hog_channels * in_channel], 
                                          se=self.se, one_conv=self.args.ssl_one_conv)
        
        # multi-order branches
        if self.mo_mae_weights:
            self.mo_mae_out0 = SSL_Output_Layer(self.stages_chs[2], self.stages_chs[2]//4, 
                len(self.args.mo_p) * in_channel, up_scale=self.args.cum_pool_scales[3], se=self.se, one_conv=self.args.ssl_one_conv)
            self.mo_mae_out1 = SSL_Output_Layer(self.stages_chs[1], self.stages_chs[1]//4, 
                len(self.args.mo_p) * in_channel, up_scale=self.args.cum_pool_scales[2], se=self.se, one_conv=self.args.ssl_one_conv)
            self.mo_mae_out2 = SSL_Output_Layer(self.stages_chs[0], self.stages_chs[0]//4, 
                len(self.args.mo_p) * in_channel, up_scale=self.args.cum_pool_scales[1], se=self.se, one_conv=self.args.ssl_one_conv)
            self.mo_mae_out = SSL_Last_Layer([self.stages_chs[1], self.stages_chs[1]//4, 8, 
                len(self.args.mo_p) * in_channel], se=self.se, one_conv=self.args.ssl_one_conv)
        
        if self.mo_hog_weights:
            self.mo_hog_out0 = SSL_Output_Layer(self.stages_chs[2], self.stages_chs[2]//4, 
                hog_channels * len(self.args.mo_p) * in_channel, up_scale=self.args.cum_pool_scales[3], se=self.se, one_conv=self.args.ssl_one_conv)
            self.mo_hog_out1 = SSL_Output_Layer(self.stages_chs[1], self.stages_chs[1]//4, 
                hog_channels * len(self.args.mo_p) * in_channel, up_scale=self.args.cum_pool_scales[2], se=self.se, one_conv=self.args.ssl_one_conv)
            self.mo_hog_out2 = SSL_Output_Layer(self.stages_chs[0], self.stages_chs[0]//2, 
                hog_channels * len(self.args.mo_p) * in_channel, up_scale=self.args.cum_pool_scales[1], se=self.se, one_conv=self.args.ssl_one_conv)
            self.mo_hog_out = SSL_Last_Layer([self.stages_chs[1], self.stages_chs[1]//4, 16, 
                hog_channels * len(self.args.mo_p) * in_channel], se=self.se, one_conv=self.args.ssl_one_conv)
        
        if self.args.he_init:
            self.init_params()
    
    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            # elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
            #     init.constant_(m.weight, 1)
            #     init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            # elif isinstance(m, nn.Parameter):
            #     init.normal_(m, std=0.001)
    
    def forward(self, x, label=None):
        
        e0_0, e0_1 = self.ec0(x)
        e1_0, e1_1 = self.ec1(e0_1)
        e2_0, e2_1 = self.ec2(e1_1)
        e3_0, e3_1 = self.ec3(e2_1)

        center = self.c(e3_1)
        att0 = self.att0(e3_1, center)
        cat0 = torch.cat((center,att0), 1)
        dc0, up0 = self.dc0(cat0)
        out0, out0_t = self.out0(dc0)

        att1 = self.att1(e2_1, up0)
        cat1 = torch.cat((up0, att1), 1)
        dc1, up1 = self.dc1(cat1)
        out1, out1_t = self.out1(dc1)

        att2 = self.att2(e1_1, up1)
        cat2 = torch.cat((up1, att2), 1)
        dc2, up2 = self.dc2(cat2)
        out2, out2_t = self.out2(dc2)

        att3 = self.att3(e0_1, up2)
        cat3 = torch.cat((up2, att3), 1)
        if self.uper:
            [out3, out3_t] = self.out(torch.cat([out0_t,out1_t,out2_t,cat3], dim=1))
        else:
            [out3, out3_t] = self.out(cat3)

        pred = torch.cat((out0,out1,out2,out3), 1)
        pred_feats = [out0_t,out1_t,out2_t,out3_t]
        
        # pdb.set_trace()
        if self.use_mae:
            mae_out0 = self.mae_out0(dc0)
            mae_out1 = self.mae_out1(dc1)
            mae_out2 = self.mae_out2(dc2)
            mae_out3 = self.mae_out(cat3)
            mae_pred = torch.cat((mae_out0,mae_out1,mae_out2,mae_out3), 1)

        if self.use_hog:
            hog_out0 = self.hog_out0(dc0)
            hog_out1 = self.hog_out1(dc1)
            hog_out2 = self.hog_out2(dc2)
            hog_out3 = self.hog_out(cat3)
            hog_pred = torch.cat((hog_out0,hog_out1,hog_out2,hog_out3), 1)
        
        if self.mo_mae_weights:
            mo_mae_out0 = self.mo_mae_out0(dc0)
            mo_mae_out1 = self.mo_mae_out1(dc1)
            mo_mae_out2 = self.mo_mae_out2(dc2)
            mo_mae_out3 = self.mo_mae_out(cat3)
            mo_mae_pred = torch.cat((mo_mae_out0,mo_mae_out1,mo_mae_out2,mo_mae_out3), 1)

        if self.mo_hog_weights:
            mo_hog_out0 = self.mo_hog_out0(dc0)
            mo_hog_out1 = self.mo_hog_out1(dc1)
            mo_hog_out2 = self.mo_hog_out2(dc2)
            mo_hog_out3 = self.mo_hog_out(cat3)
            mo_hog_pred = torch.cat((mo_hog_out0,mo_hog_out1,mo_hog_out2,mo_hog_out3), 1)
        
        # weighted voting with different stage outputs during test
        # if len(self.vote_weights) and (not self.training):
        if len(self.vote_weights):
            assert(self.args.ds_up_conv), 'Wrong setting for the args.vote_weights and args.ds_up_conv'
            out3_vote = 0
            for vi in range(len(self.vote_weights)):
                out3_vote += pred[:,-(vi+1),...] * self.vote_weights[-(vi+1)]
            out3_vote = out3_vote.unsqueeze(1)
            pred = torch.cat((out0,out1,out2,out3,out3_vote), 1)
        
        final_outs = [None] * 13
        final_outs[0] = pred
        if self.use_mae:
            final_outs[1] = mae_pred
        if self.use_hog:
            final_outs[2] = hog_pred
        if self.mo_mae_weights:
            final_outs[3] = mo_mae_pred
        if self.mo_hog_weights:
            final_outs[4] = mo_hog_pred
        if self.args.opl:
            final_outs[5] = pred_feats
        return final_outs


def get_model():
    net = FuzzyAttention_3DUNet()
    return net

if __name__ == '__main__':
    use_gpu = True
    from torchinfo import summary
    net = get_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = net.to(device)
    summary(model, (1,1, 128, 96, 144))

