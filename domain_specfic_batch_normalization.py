from torch import nn

# Domain-Specific Batch Normalization for Unsupervised Domain Adaptation
# https://arxiv.org/abs/1906.03950
class DomainSpecificBatchNormalization(nn.Module):
    def __init__(self, num_features, num_classes, epsilon = 1e-5,
            momentum = 1e-1, affine = False, track_running_stats = True):
        super(DomainSpecificBatchNormalization, self).__init__()
        self.bns = nn.ModuleList([nn.BatchNorm2d(num_features, epsilon,
            momentum, affine, track_running_stats) for _ in range(num_classes)])

    def reset_running_stats(self):
        for bn in self.bns:
            bn.reset_running_stats()

    def reset_parameters(self):
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, y):
        bn = self.bns[y[0]]
        return bn(x)
