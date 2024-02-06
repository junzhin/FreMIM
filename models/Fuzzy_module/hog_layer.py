# Copyright (c) OpenMMLab. All rights reserved.
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb


class HOGLayerC(nn.Module):
    """Generate hog feature for each batch images. This module is used in
    Maskfeat to generate hog feature. This code is borrowed from.

    <https://github.com/facebookresearch/SlowFast/blob/main/slowfast/models/operators.py>
    Args:
        nbins (int): Number of bin. Defaults to 9.
        pool (float): Number of cell. Defaults to 8.
        gaussian_window (int): Size of gaussian kernel. Defaults to 16.
    """

    def __init__(self,
                 nbins: int = 9,
                 pool: int = 8,
                 gaussian_window: int = 16,
                 num_modalities: int = 1) -> None: # note, num_modalities is deemed as out_channels.
        super().__init__()
        self.nbins = nbins
        self.pool = pool
        self.num_modalities = num_modalities
        self.pi = math.pi
        weight_x = torch.FloatTensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        weight_x = weight_x.view(1, 1, 3, 3).repeat(self.num_modalities, 1, 1, 1)
        weight_y = weight_x.transpose(2, 3)
        self.register_buffer('weight_x', weight_x)
        self.register_buffer('weight_y', weight_y)

        self.gaussian_window = gaussian_window
        if gaussian_window:
            gkern = self.get_gkern(gaussian_window, gaussian_window // 2)
            self.register_buffer('gkern', gkern)

    def get_gkern(self, kernlen: int, std: int) -> torch.Tensor:
        """Returns a 2D Gaussian kernel array."""

        def _gaussian_fn(kernlen: int, std: int) -> torch.Tensor:
            n = torch.arange(0, kernlen).float()
            n -= n.mean()
            n /= std
            w = torch.exp(-0.5 * n**2)
            return w

        gkern1d = _gaussian_fn(kernlen, std)
        gkern2d = gkern1d[:, None] * gkern1d[None, :]
        return gkern2d / gkern2d.sum()

    def _reshape(self, hog_feat: torch.Tensor) -> torch.Tensor:
        hog_feat = hog_feat.flatten(1, 2)
        unfold_size = hog_feat.shape[-1] // 14
        hog_feat = (
            hog_feat.permute(0, 2, 3,
                             1).unfold(1, unfold_size, unfold_size).unfold(
                                 2, unfold_size,
                                 unfold_size).flatten(1, 2).flatten(2))
        return hog_feat

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Generate hog feature for each batch images.

        Args:
            x (torch.Tensor): Input images of shape (N, num_modalities, H, W).
        Returns:
            torch.Tensor: Hog features.
        """
        # pdb.set_trace()
        # input is RGB or medical image with shape [B 3 H W] or [N, num_modalities, H, W]
        x = F.pad(x, pad=(1, 1, 1, 1), mode='reflect')
        gx_rgb = F.conv2d(
            x, self.weight_x, bias=None, stride=1, padding=0, groups=self.num_modalities)
        gy_rgb = F.conv2d(
            x, self.weight_y, bias=None, stride=1, padding=0, groups=self.num_modalities)
        norm_rgb = torch.stack([gx_rgb, gy_rgb], dim=-1).norm(dim=-1)
        phase = torch.atan2(gx_rgb, gy_rgb)
        phase = phase / self.pi * self.nbins  # [min_value, max_value] = [-9, 9]

        b, c, h, w = norm_rgb.shape
        out = torch.zeros((b, c, self.nbins, h, w),
                          dtype=torch.float,
                          device=x.device)
        phase = phase.view(b, c, 1, h, w)
        norm_rgb = norm_rgb.view(b, c, 1, h, w)
        if self.gaussian_window:
            if h != self.gaussian_window or w != self.gaussian_window:
                assert (h % self.gaussian_window == 0 and w % self.gaussian_window == 0), 'h {} w {} gw {}'.format(
                    h, w, self.gaussian_window)
                repeat_rate_h = h // self.gaussian_window
                repeat_rate_w = w // self.gaussian_window
                temp_gkern = self.gkern.repeat([repeat_rate_h, repeat_rate_w])
            else:
                temp_gkern = self.gkern
            norm_rgb *= temp_gkern

        out.scatter_add_(2, phase.floor().long() % self.nbins, norm_rgb)

        # out = out.unfold(3, self.pool, self.pool)
        # out = out.unfold(4, self.pool, self.pool)
        # out = out.sum(dim=[-1, -2])

        # out = F.normalize(out, p=2, dim=2) # torch.Size([256, 1, 9, 96, 96])

        # return self._reshape(out) # torch.Size([256, 196, 108])
        return out
