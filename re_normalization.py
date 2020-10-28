import torch
from torch import nn

# Batch Renormalization: Towards Reducing Minibatch Dependence in Batch-Normali-
# zed Models
# https://arxiv.org/abs/1702.03275
class ReNormalization(nn.Module):
    def __init__(self, num_features, epsilon = 1e-4, momentum = 0.9,
            affine = True):
        super(ReNormalization, self).__init__()
        self.affine = affine
        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(num_features))
            self.bias = nn.Parameter(torch.Tensor(num_features))
        self.register_buffer('running_var', torch.ones(1, num_features, 1, 1))
        self.register_buffer('running_mean', torch.ones(1, num_features, 1, 1))
        self.register_buffer('num_tracked_batches', torch.tensor(0))
        self.epsilon = epsilon
        self.momentum = momentum

    @property
    def r_max(self):
        return (2 / 35000 * self.num_tracked_batches + 25 / 35).clamp_(1.0, 3.0)

    @property
    def d_max(self):
        return (5 / 20000 * self.num_tracked_batches - 25 / 20).clamp_(0.0, 5.0)

    def update_stats(self, x):
        mean = x.mean((0, 2, 3), keepdim = True)
        var = x.var((0, 2, 3), keepdim = True)
        std = x.std((0, 2, 3), keepdim = True, unbiased = False)
        running_std = (self.running_var + self.epsilon).sqrt()
        r = (std / running_std).clamp_(1 / self.r_max, self.r_max).detach()
        d = (mean / self.running_mean).clamp_(-self.d_max, self.d_max).detach()
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
        self.num_tracked_batches += 1
        return x
