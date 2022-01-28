import torch
from torch import nn


class ReconstructModel(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, seq_len, dropout):
        super(ReconstructModel, self).__init__()
        self.in_dim = in_dim
        self.seq_len = seq_len

        self.gru = nn.GRU(in_dim, hid_dim, num_layers=1, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hid_dim, out_dim)

    def forward(self, hidden):
        output = []
        temp_input = torch.zeros((hidden.shape[1], 1, self.in_dim), dtype=torch.float).to(hidden.device)
        for _ in range(self.seq_len):
            temp_input, hidden = self.gru(temp_input, hidden)
            temp_input = self.fc(temp_input)
            output.append(temp_input)

        # 翻转
        inv_idx = torch.arange(self.seq_len - 1, -1, -1).long()
        output = torch.cat(output, dim=1)[:, inv_idx, :]
        return output
