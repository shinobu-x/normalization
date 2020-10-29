from torch import nn

# A Learned Representation For Artistic Style
# https://arxiv.org/abs/1610.07629v5
class ConditionalInstanceNormalization(nn.Module):
    def __init__(self, num_features, num_classes, bias = False, affine = False,
            track_running_stats = False):
        super().__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.bias = bias
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.instance_norm = nn.InstanceNorm2d(self.num_features,
                affine = self.affine,
                track_running_stats = self.track_running_stats)
        if self.bias:
            self.embed = nn.Embedding(self.num_classes, self.num_features * 2)
            self.embed.weight.data[:, : self.num_features].uniform_()
            self.embed.weight.data[:, self.num_features].zero_()
        else:
            self.embed = nn.Embedding(self.num_classes, self.num_features)
            self.embed.weight.data.uniform_()

    def forward(self, x, y):
        x = self.instance_norm(x)
        if self.bias:
            gamma, beta = self.embed(y).chunk(2, -1)
            z = gamma.view(-1, self.num_features, 1, 1) * x + beta.view(-1,
                    self.num_features, 1, 1)
        else:
            gamma = self.embed(y)
            z = gamma.view(-1, self.num_features, 1, 1) * x
        return z
