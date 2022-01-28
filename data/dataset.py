from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, data, w=64):
        self.data = data
        self.w = w

    def __getitem__(self, index):
        x = self.data[index:index + self.w]
        y = self.data[index + self.w:index + self.w + 1]

        return x, y

    def __len__(self):
        return len(self.data) - self.w
