import torch
from torch import nn

# Batch Renormalization: Towards Reducing Minibatch Dependence in Batch-Normali-
# zed Models
# https://arxiv.org/abs/1702.03275
class ReNormalization(nn.Module):
    def __init__(self, num_features, r_max = 1, d_max = 0, epsilon = 1e-4,
            momentum = 0.9, affine = True):
        super(ReNormalization, self).__init__()
        self.affine = affine
        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(num_features))
            self.bias = nn.Parameter(torch.Tensor(num_features))
        self.register_buffer('running_var', torch.ones(1, num_features, 1, 1))
        self.register_buffer('running_mean', torch.ones(1, num_features, 1, 1))
        self.r_max = r_max
        self.d_max = d_max
        self.epsilon = epsilon
        self.momentum = momentum

    def update_stats(self, x):
        mean = x.mean((0, 2, 3), keepdim = True)
        var = x.var((0, 2, 3), keepdim = True)
        std = (var + self.epsilon).sqrt()
        running_std = (self.running_var + self.epsilon).sqrt()
        r = torch.clamp(std / running_std, min = 1 / self.r_max,
                max = self.r_max).detach()
        d = torch.clamp((mean / self.running_mean) / running_std,
                min = -self.d_max, max = self.d_max).detach()
        self.running_mean.lerp_(mean, self.momentum)
        self.running_var.lerp_(var, self.momentum)
        return mean, std, r, d

    def forward(self, x):
        if self.training:
            with torch.no_grad():
                mean, std, r, d = self.update_stats(x)
            x = (x - mean) / std * r + d
        else:
            x = (x - self.running_mean) / \
                    (self.running_var + self.epsilon).sqrt()
        if self.affine:
            return self.weight * x.transpose(1, -1) + self.bias
        return x
