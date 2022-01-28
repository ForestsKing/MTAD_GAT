import torch
from torch import nn

from model.conv import ConvLayer
from model.forecast import ForecastModel
from model.gat import FeatureAttentionLayer, TemporalAttentionLayer
from model.gru import GRULayer
from model.reconstruct import ReconstructModel


class MTAD_GAT(nn.Module):
    def __init__(self, n_features=30, gat_h_dim=128, gru_h_dim=128, rec_h_dim=128, for_h_dim=128, seq_len=64,
                 dropout=0.5):
        super(MTAD_GAT, self).__init__()
        self.conv = ConvLayer(n_features=n_features)

        self.feature_gat = FeatureAttentionLayer(in_features=seq_len, h_dim=gat_h_dim, out_features=seq_len,
                                                 dropout=dropout)
        self.temporal_gat = TemporalAttentionLayer(in_features=n_features, h_dim=gat_h_dim, out_features=n_features,
                                                   dropout=dropout)

        self.gru = GRULayer(in_dim=3 * n_features, hid_dim=gru_h_dim, dropout=dropout)

        self.reconstruct_model = ReconstructModel(in_dim=n_features, hid_dim=rec_h_dim, out_dim=n_features,
                                                  seq_len=seq_len, dropout=dropout)
        self.forest_model = ForecastModel(in_dim=gru_h_dim, hid_dim=for_h_dim, out_dim=n_features)

    def forward(self, x):
        x = self.conv(x)

        hf = self.feature_gat(x)
        ht = self.temporal_gat(x)

        h = torch.cat([x, hf, ht], dim=2)
        h = self.gru(h)

        reconstruct = self.reconstruct_model(h)
        forecast = self.forest_model(h)
        return reconstruct, forecast
