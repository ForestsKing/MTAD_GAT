import torch.nn.functional as F
from torch import nn


class ForecastModel(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout):
        super(ForecastModel, self).__init__()
        self.dropout = dropout
        self.layers = nn.Sequential()
        self.layers.add_module('layer1', nn.Linear(in_dim, hid_dim))
        for i in range(3):
            self.layers.add_module('layer' + str(i + 2), nn.Linear(hid_dim, hid_dim))

        self.fc = nn.Linear(hid_dim, out_dim)
        self.norm = nn.LayerNorm(hid_dim)

    def forward(self, x):
        x = x.transpose(0, 1).reshape(x.shape[1], -1)
        x = self.layers(x)
        x = F.relu(x)
        x = self.norm(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.fc(x)
        return x.unsqueeze(1)
