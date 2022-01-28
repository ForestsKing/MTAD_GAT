from torch import nn


class GRULayer(nn.Module):
    def __init__(self, in_dim, hid_dim, dropout):
        super(GRULayer, self).__init__()
        self.gru = nn.GRU(in_dim, hid_dim, num_layers=1, batch_first=True, dropout=dropout)

    def forward(self, x):
        _, hidden = self.gru(x)
        return hidden
