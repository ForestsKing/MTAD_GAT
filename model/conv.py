import torch.nn.functional as F
from torch import nn


class ConvLayer(nn.Module):
    def __init__(self, n_features):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv1d(in_channels=n_features, out_channels=n_features, kernel_size=(7,), padding=(3,))
        self.norm = nn.BatchNorm1d(n_features)

    def forward(self, x):
        x = x.transpose(-2, -1)
        x = self.conv(x)
        x = self.norm(x)
        x = F.relu(x)
        return x.transpose(-2, -1)
