import numpy as np
from torch.utils.data import Dataset


def definir_salida(x, y):
    if (x - 0.25)**2 + (y - 0.25)**2 < 0.25**2:
        return 1
    if (x - 0.75)**2 + (y - 0.25)**2 < 0.25**2:
        return 2
    if (x - 0.25)**2 + (y - 0.75)**2 < 0.25**2:
        return 3
    if (x - 0.75)**2 + (y - 0.75)**2 < 0.25**2:
        return 4
    return 0


def calcula_class_weights(x):
    return 1. / np.unique(x, return_counts=True)[1]


class MiDataset(Dataset):
    def __init__(self, data):
        super(MiDataset, self).__init__()
        self.data = data
    
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index, :-1], self.data[index, -1]
