import torch
import torch.nn.functional as F
from torch import nn

# Instance Enhancement Batch Normalization: an Adaptive Regulator of Batch Noise
# https://arxiv.org/abs/1908.04008
class InstanceEnhancementNormalization(nn.BatchNorm2d):
    def __init__(self, num_features, epsilon = 1e-4, momentum = 0.9,
            affine = False, training = False):
        super(InstanceEnhancementNormalization, self).__init__(num_features,
                epsilon, momentum, affine)
        self.num_features = num_features
        self.epsilon = epsilon
        self.momentum = momentum
        self.affine = affine
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.sigmoid = nn.Sigmoid()
        self.training = training
        self.weight = nn.Parameter(
                torch.Tensor(1, self.num_features, 1, 1))#.data.fill_(1)
        self.bias = nn.Parameter(
                torch.Tensor(1, self.num_features, 1, 1))#.data.fill_(0)
        self.gamma_hat = nn.Parameter(
                torch.Tensor(1, self.num_features, 1, 1)).data.fill_(0)
        self.beta_hat = nn.Parameter(
                torch.Tensor(1, self.num_features, 1, 1)).data.fill_(-1)

    def forward(self, x):
        # \delta_{bc} = sig(\hat{\gamma_c} * m_{bc} + \hat{beta_c})
        delta = self.sigmoid(self.pool(x) * self.gamma_hat + self.beta_hat)
        x_hat = F.batch_norm(x, self.running_mean, self.running_var, None, None,
                self.training, self.momentum, self.epsilon)
        # Y_{bc} = \hat{X_{bc}} * (\gamma_c * \delta_{bc}) + \beta_c
        y = x_hat * self.weight * delta + self.bias
        return y
