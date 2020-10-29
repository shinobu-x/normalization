import torch
from torch import nn

# Modulating early visual processing by language
# https://arxiv.org/abs/1707.00683
class ConditionalBatchNormalization(nn.Module):
    def __init__(self, num_features, num_classes, bias = True, affine = True):
        super().__init__()
        self.num_features = num_features
        self.bias = bias
        self.bn = nn.BatchNorm2d(num_features, affine = affine)
        if self.bias:
            self.embed = nn.Embedding(num_classes, num_features * 2)
            self.embed.weight.data[:, : num_features].uniform_()
            self.embed.weight.data[:, num_features].zero_()
        else:
            self.embed = nn.Embedding(num_classes, num_features)
            self.embed.weight.data.uniform_()

    def forward(self, x, y):
        x = self.bn(x)
        if self.bias:
            gamma, beta = self.embed(y).chunk(2, 1)
            x = gamma.view(-1, self.num_features, 1, 1) * x + beta.view(
                    -1, self.num_features, 1, 1)
        else:
            gamma = self.embed(y)
            x = gamma.view(-1, self.num_features, 1, 1) * x
        return x
