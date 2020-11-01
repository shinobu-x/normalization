import torch
import torch.nn.functional as F
from torch import nn

# Switchable Normalization for Learning-to-Normalize Deep Representation
# https://arxiv.org/abs/1907.10473
class SwitchableNormalization(nn.Module):
    def __init__(self, num_features, epsilon = 1e-5, momentum = 0.9,
            enable_moving_average = False, enable_bn = False,
            enable_last_gamma = False, learnable = False):
        super(SwitchableNormalization, self).__init__()
        self.num_features = num_features
        self.epsilon = epsilon
        self.momentum = momentum
        self.enable_moving_average = enable_moving_average
        self.enable_bn = enable_bn
        self.enable_last_gamma = enable_last_gamma
        self.learnable = learnable
        self.weight = nn.Parameter(torch.ones(1, self.num_features, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, self.num_features, 1, 1))
        if self.enable_bn:
            self.mean_weight = self.var_weight = nn.Parameter(torch.ones(3))
            self.register_buffer('running_mean', torch.zeros(1,
                self.num_features, 1))
            self.register_buffer('running_var', torch.zeros(1,
                self.num_features, 1))
        else:
            self.mean_weight = self.var_weight = nn.Parameter(torch.ones(2))
        self.reset_parameters()

    def reset_parameters(self):
        if self.enable_bn:
            self.running_mean.zero_()
            self.running_var.zero_()
        if self.enable_last_gamma:
            self.weight.data.fill_(0)
        else:
            self.weight.data.fill_(1)
        self.bias.data.zero_()

    def forward(self, x):
        n, c, h, w = x.size()
        x = x.view(n, c, -1)
        # \mu_{in} = 1/HW * \sum^{H,W}_{i,j} h_{ncij}
        mu_in = x.mean(-1, keepdim = True)
        # \sigma^2_{in} = 1/HW * \sum^{H,W}_{i,j} (h_{ncij} - mu_{in}) ** 2
        var_in = x.var(-1, keepdim = True)
        # mu_{ln} = 1/C * \sum^C_{c=1} mu_{in}
        mu_ln = mu_in.mean(1, keepdim = True)
        # \sigma^2_{ln} = 1/C * \sum^C_{c=1} (\sigma^2_{in} + \mu^2_{in}) -
        # \mu^2_{ln}
        t = var_in + mu_in ** 2
        var_ln = (var_in + mu_in ** 2).mean(1, keepdim = True) - mu_ln ** 2
        if self.enable_bn:
            if self.learnable:
                # \mu_{bn} = 1/N * \sum^N_{n=1} \mu_{in}
                mu = mu_in.mean(0, True)
                # \sigma^2_{bn} = 1/N \sum^N_{n=1} (\sigma^2_{in} + \mu^2_{in})
                # - \mu^2_{bn}
                var = (var_in + mu_in ** 2).mean(0, keepdim = True) - mu ** 2
                if self.enable_moving_average:
                    self.running_mean.mul_(self.momentum)
                    self.running_mean.add_((1 - self.momentum) * mu.data)
                    self.running_var.mul_(self.momentum)
                    self.running_var.add_((1 - self.momentum) * var.data)
                else:
                    self.running_mean.add_(mu.data)
                    self.running_var.add_(var.data + mu.data ** 2)
            else:
                mu = torch.autograd.Variable(self.running_mean)
                var = torch.autograd.Variable(self.running_var)
        s = nn.Softmax(0)
        mu_weight = s(self.mean_weight)
        var_weight = s(self.var_weight)
        if self.enable_bn:
            mu = mu_weight[0] * mu_in + mu_weight[1] * mu_ln + \
                    mu_weight[2] * mu
            var = var_weight[0] * var_in + var_weight[1] * var_ln + \
                    var_weight[2] * var_in
        else:
            mu = mu_weight[0] * mu_in + mu_weight[1] * mu_ln
            var = var_weight[0] * var_in + var_weight[1] * var_ln
        x = (x - mu) / (var + self.epsilon).sqrt()
        x = x.view(n, c, h, w)
        a = x * self.weight
        b = a + self.bias
        return x * self.weight + self.bias
