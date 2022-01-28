import torch
import torch.nn.functional as F
from torch import nn


class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, n_head=8, dropout=0.2, alpha=0.2, last=False):
        super(GATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_head = n_head
        self.dropout = dropout
        self.alpha = alpha
        self.last = last

        Ws = []
        As = []
        for _ in range(self.n_head):
            W = nn.Linear(in_features, out_features)
            A = nn.Linear(2 * out_features, 1)
            nn.init.xavier_uniform_(W.weight, gain=1.414)
            nn.init.xavier_uniform_(A.weight, gain=1.414)
            Ws.append(W)
            As.append(A)
        self.Ws = torch.nn.ModuleList(Ws)
        self.As = torch.nn.ModuleList(As)

    def forward(self, h):
        outs = []
        for i in range(self.n_head):
            W = self.Ws[i]
            A = self.As[i]

            Wh = W(h)
            Whi = Wh.unsqueeze(2).repeat(1, 1, Wh.size(1), 1)
            Whj = Wh.unsqueeze(1).repeat(1, Wh.size(1), 1, 1)
            Whi_cat_Whj = torch.cat((Whi, Whj), dim=3)

            e = A(Whi_cat_Whj).squeeze()
            e = F.leaky_relu(e, negative_slope=self.alpha)

            a = F.softmax(e, dim=-1)
            a = F.dropout(a, self.dropout, training=self.training)

            aWh = torch.matmul(a, Wh)
            if self.last:
                out = aWh
            else:
                out = F.elu(aWh)
            outs.append(out)

        if self.last:
            outs = torch.mean(torch.stack(outs, dim=3), dim=3)
        else:
            outs = torch.cat(outs, dim=2)
        return F.sigmoid(outs)


class FeatureAttentionLayer(nn.Module):
    def __init__(self, in_features, h_dim, out_features, n_head=8, dropout=0.2):
        super(FeatureAttentionLayer, self).__init__()
        self.conv1 = GATLayer(in_features, h_dim, n_head=n_head, dropout=dropout, last=False)
        self.conv2 = GATLayer(h_dim * n_head, out_features, n_head=n_head, dropout=dropout, last=True)

    def forward(self, x):
        x = x.transpose(-2, -1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x.transpose(-2, -1)


class TemporalAttentionLayer(nn.Module):
    def __init__(self, in_features, h_dim, out_features, n_head=8, dropout=0.2):
        super(TemporalAttentionLayer, self).__init__()
        self.conv1 = GATLayer(in_features, h_dim, n_head=n_head, dropout=dropout, last=False)
        self.conv2 = GATLayer(h_dim * n_head, out_features, n_head=n_head, dropout=dropout, last=True)

    def forward(self, x):
        x = x
        x = self.conv1(x)
        x = self.conv2(x)
        return x
