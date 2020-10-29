import torch
from torch import nn

# Filter Response Normalization Layer: Eliminating Batch Dependence in the Trai-
# ning of Deep Neural Networks
# https://arxiv.org/abs/1911.09737
class ThreshholdedLinearUnit(nn.Module):
    def __init__(self, num_features):
        super(ThreshholdedLinearUnit, self).__init__()
        self.num_features = num_features
        self.tau = nn.init.zeros_(nn.Parameter(torch.Tensor(1, num_features, 1,
            1), requires_grad = True))

    def reset_parameter(self):
        return nn.init.zeros_(self.tau)

    def forward(self, x):
        # z_i = max(y_i, \tau)
        return torch.max(x, self.tau)

class FilterResponseNormalization(nn.Module):
    def __init__(self, num_features, epsilon = 1e-6, is_learnable = False):
        super(FilterResponseNormalization, self).__init__()
        self.num_features = num_features
        self.constant_epsilon = epsilon
        self.is_learnable = is_learnable
        self.gamma = nn.parameter.Parameter(torch.Tensor(1, num_features, 1,
            1), requires_grad = True)
        self.beta = nn.parameter.Parameter(torch.Tensor(1, num_features, 1, 1),
            requires_grad = True)
        self.tlu = ThreshholdedLinearUnit(self.num_features)
        if is_learnable:
            self.epsilon = nn.parameter.Parameter(torch.Tensor(1),
                    requires_grad = True)
        else:
            self.register_buffer('epsilon', torch.Tensor([epsilon]))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.gamma)
        nn.init.zeros_(self.beta)
        if self.is_learnable:
            nn.init.constant_(self.epsilon, self.constant_epsilon)

    def forward(self, x):
        # \nu^2 = \sum_i x^2_i / N
        nu = x.pow(2).mean([2, 3], keepdim = True)
        # y_i = \gamma * x_i / \sqrt(\nu^2 + \epsilon) + \beta
        x = x * torch.rsqrt(nu + self.epsilon.abs())
        x = self.gamma * x + self.beta
        return self.tlu(x)
