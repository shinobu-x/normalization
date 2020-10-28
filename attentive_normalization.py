import torch
from torch import nn
from torch import sigmoid

# Attentive Normalization
# https://arxiv.org/abs/1908.01259
class AttentiveNormalization(nn.BatchNorm2d):
    def __init__(self, num_features, hidden_channels = None, epsilon = 1e-5,
            momentum = 0.9, track_running_stats = True):
        super(AttentiveNormalization, self).__init__(
                num_features = num_features, eps = epsilon,
                momentum = momentum, affine = True,
                track_running_stats = track_running_stats)
        self.gamma = nn.Parameter(torch.Tensor(hidden_channels, num_features))
        self.beta = nn.Parameter(torch.Tensor(hidden_channels, num_features))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.l = nn.Linear(num_features, hidden_channels)

    def forward(self, x):
        x = super(AttentiveNormalization, self).forward(x)
        shape = x.size()
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.l(y)
        y = sigmoid(y)
        gamma = (y @ self.gamma).unsqueeze(-1).unsqueeze(-1).expand(shape)
        beta = (y @ self.beta).unsqueeze(-1).unsqueeze(-1).expand(shape)
        return x * gamma + beta
