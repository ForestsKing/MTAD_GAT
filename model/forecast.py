from torch import nn


class ForecastModel(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super(ForecastModel, self).__init__()
        self.layers = nn.Sequential()
        self.layers.add_module('layer1', nn.Linear(in_dim, hid_dim))
        for i in range(1):
            self.layers.add_module('layer' + str(i + 2), nn.Linear(hid_dim, hid_dim))
        self.fc = nn.Linear(hid_dim, out_dim)

    def forward(self, x):
        x = x.transpose(0, 1).reshape(x.shape[1], -1)
        x = self.layers(x)
        x = self.fc(x)
        return x.unsqueeze(1)
