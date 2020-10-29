from torch import nn
from torch.nn import functional as F

# Micro-Batch Training with Batch-Channel Normalization and Weight Standardizat-
# ion
# https://arxiv.org/abs/1903.10520
class WeightStandardization(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1,
            padding = 0, dilation = 1, groups = 1, bias = True):
        super(WeightStandardization, self).__init__(in_channels, out_channels,
                kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight
        mean = weight.mean(1, True).mean(2, True).mean(3, True)
        weight = weight - mean
        std = weight.view(weight.size(0), -1).std(1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        x = F.conv2d(x, weight, self.bias, self.stride, self.padding,
                self.dilation, self.groups)
        return x
