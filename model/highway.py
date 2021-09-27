""" Ref: Srivastava, Rupesh Kumar, Klaus Greff, and Jürgen Schmidhuber. "Highway networks." arXiv preprint arXiv:1505.00387 (2015).
"""

from torch import nn

class Highway(nn.Module):

    def __init__(self, in_features, out_features):

        super(Highway, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.transform_gate = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.Sigmoid()
        )

        if in_features == out_features:
            self.linear_transform = None
        else:
            self.linear_transform = nn.Linear(in_features, out_features, bias=False)

        self.fc_layer = nn.Sequential(  # normal fully connected layer
            nn.Linear(in_features, out_features),
            nn.ReLU()
        )

        self._init_weight()

    def _init_weight(self):

        self.transform_gate[0].bias.data.fill_(-1)

    def forward(self, x):

        gate = self.transform_gate(x)

        if self.in_features == self.out_features:
            y = gate * self.fc_layer(x) + (1 - gate) * x
        else:
            y = gate * self.fc_layer(x) + (1 - gate) * self.linear_transform(x)

        return y
