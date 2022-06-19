from nlp100pon_70 import LoadData
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(MLP, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False) # バイアス入れない

    def forward(self, x):
        x = self.linear(x)
        return F.softmax(x, dim=1)


if __name__ == "__main__":

    path = '../../data/NewsAggregatorDataset/chap08_avgembed.pickle'
    
    X, Y = LoadData(path)
    
    X_train, y_train = torch.from_numpy(X['train'].astype(np.float32)), torch.tensor(Y['train'])
    model = MLP(in_dim=300, out_dim=4)
    print('正解ラベル:\t', y_train[:4])
    print('予測結果:\t', model.forward(X_train[:4]).argmax(dim=1))
    
"""
正解ラベル:      tensor([0, 2, 0, 2])
予測結果:        tensor([3, 3, 3, 3])
"""

