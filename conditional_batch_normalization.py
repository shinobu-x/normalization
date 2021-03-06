from torch import nn

# Modulating early visual processing by language
# https://arxiv.org/abs/1707.00683
class ConditionalBatchNormalization(nn.Module):
    def __init__(self, num_features, num_classes, bias = False, affine = False):
        super().__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.bias = bias
        self.affine = affine
        self.bn = nn.BatchNorm2d(self.num_features, affine = self.affine)
        if self.bias:
            self.embed = nn.Embedding(self.num_classes, self.num_features * 2)
            self.embed.weight.data[:, : self.num_features].uniform_()
            self.embed.weight.data[:, self.num_features].zero_()
        else:
            self.embed = nn.Embedding(self.num_classes, self.num_features)
            self.embed.weight.data.uniform_()

    def forward(self, x, y):
        x = self.bn(x)
        if self.bias:
            gamma, beta = self.embed(y).chunk(2, 1)
            z = gamma.view(-1, self.num_features, 1, 1) * x + beta.view(
                    -1, self.num_features, 1, 1)
        else:
            gamma = self.embed(y)
            z = gamma.view(-1, self.num_features, 1, 1) * x
        return z
