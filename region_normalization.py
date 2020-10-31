import torch
import torch.nn.functional as F
from torch import nn

# Region Normalization for Image Inpainting
# https://arxiv.org/abs/1911.10375
class RegionNormalization(nn.Module):
    def __init__(self, num_features, affine = False,
            track_running_stats = False, learnable = False):
        super(RegionNormalization, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.learnable = learnable
        if self.learnable:
            self.norm = RegionNormalization(self.num_features)
            self.foreground_gamma = nn.Parameter(torch.zeros(num_features),
                    requires_grad = True)
            self.foreground_beta = nn.Parameter(torch.zeros(num_features),
                    requires_grad = True)
            self.background_gamma = nn.Parameter(torch.zeros(num_features),
                    requires_grad = True)
            self.background_beta = nn.Parameter(torch.zeros(num_features),
                    requires_grad = True)
        else:
            self.norm = nn.BatchNorm2d(num_features, affine,
                    track_running_stats)

    def forward(self, x, mask):
        if self.learnable:
            mask = F.interpolate(mask, size = x.size()[2:], mode = 'nearest')
            x = self.norm(x, mask)
            foreground = (x * mask) * (1 +
                    self.foreground_gamma[None, :, None, None]
                    ) + self.foreground_beta[None, :, None, None]
            background = (x * (1 - mask)) * (1 +
                    self.background_gamma[None, :, None, None]
                    ) + self.background_beta[None, :, None, None]
        else:
            mask = mask.detach()
            foreground = self.compute_region_norm(x * mask, mask)
            background = self.compute_region_norm(x * (1 - mask), 1 - mask)
        return foreground + background

    def compute_region_norm(self, region, mask):
        shape = region.size()
        sum = torch.sum(region, [0, 2, 3])
        sr = torch.sum(mask, [0, 2, 3])
        sr[sr == 0] = 1
        mu = sum /sr
        x = region + (1 - mask) * mu[None, :, None, None] * (torch.sqrt(
            sr / (shape[0] * shape[2] * shape[3])))[None, :, None, None]
        return self.norm(x)
