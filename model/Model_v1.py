import torch.nn as nn
from .utils import BACKBONE

class Model_v1(nn.Module):
    def __init__(self, n_class, backbone, n_fc = 1, drop = 0.1):
        super().__init__()
        print('test')
        network = BACKBONE.get(backbone, None)
        assert network != None
        self.network = network['model'](pretrained=True)
        in_features = self.network.fc.in_features
        self.fc = nn.Sequential()
        for i in range(n_fc - 1):
            self.fc.add_module(f'fc{i}', nn.Sequential(
                nn.Linear(in_features, int(in_features/2)),
                nn.ReLU()
            ))
            in_features = int(in_features / 2)
        self.fc.add_module(f'fc{n_fc-1}', nn.Linear(in_features, n_class))
        self.name = f'{backbone}_{n_fc}_{drop}'
    def forward(self, X):
        out = self.network(X)
        return out
    