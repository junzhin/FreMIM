import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pdb

class PPM3D(nn.ModuleList):
    """Pooling Pyramid Module used in PSPNet.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module.
        in_channels (int): Input channels.
        out_channels (int): Channels after modules, before conv_seg.
        align_corners (bool): align_corners argument of F.interpolate.
    """

    def __init__(self, pool_scales, in_channels, out_channels, in_sizes, 
                 kernel_size=3, stride=1, padding=1, 
                 align_corners=True, sc=False, **kwargs):
        super(PPM3D, self).__init__()
        self.pool_scales = pool_scales
        self.align_corners = align_corners
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_sizes = in_sizes
        assert(len(in_sizes)==3), 'Wrong size for the PPM3D module !'
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.sc = sc
        for pool_scale in pool_scales:
            self.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool3d([max(int(math.ceil(in_sizes[0]/float(pool_scale))), 1), 
                                          max(int(math.ceil(in_sizes[1]/float(pool_scale))), 1), 
                                          max(int(math.ceil(in_sizes[2]/float(pool_scale))), 1)]), 
                    nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, 
                              stride=stride, padding=padding), 
                    nn.InstanceNorm3d(out_channels), 
                    nn.LeakyReLU(inplace=True)
                    ))

    def forward(self, x, dummy_tensor=None):
        """Forward function."""
        # pdb.set_trace()
        ppm_outs = []
        for ppm in self:
            ppm_out = ppm(x)
            upsampled_ppm_out = F.interpolate(
                ppm_out, x.size()[2:], mode='trilinear', 
                align_corners=self.align_corners)
            if self.sc:
                ppm_outs.append(upsampled_ppm_out + x)
            else:
                ppm_outs.append(upsampled_ppm_out)
        ppm_outs = torch.cat(ppm_outs, 1)
        return ppm_outs
