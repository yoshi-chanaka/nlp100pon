from nlp100pon_70 import LoadData
from nlp100pon_71 import MLP
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import time

"""
参考: 
https://soroban.highreso.jp/blog/blog-2326/#CPUGPUPyTorch
https://www.mattari-benkyo-note.com/2021/03/21/pytorch-cuda-time-measurement/
"""


def MesureTimeWithCPU(dataset, batch_size, epochs=1):

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    size = len(dataloader.dataset)

    model = MLP(in_dim=300, out_dim=4)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.99)

    start = time.time()
    for epoch in range(epochs):
        
        model.train()
        for X, y in dataloader:

            # X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    return time.time() - start


if __name__ == "__main__":

    path = '../../data/NewsAggregatorDataset/chap08_avgembed.pickle'
    X, Y = LoadData(path)    
    X_train, y_train    = torch.from_numpy(X['train'].astype(np.float32)), torch.tensor(Y['train'])
    X_valid, y_valid    = torch.from_numpy(X['valid'].astype(np.float32)), torch.tensor(Y['valid'])
    X_test, y_test      = torch.from_numpy(X['test'].astype(np.float32)), torch.tensor(Y['test'])

    dataset = {}
    dataset['train']    = torch.utils.data.TensorDataset(X_train, y_train)
    dataset['valid']    = torch.utils.data.TensorDataset(X_valid, y_valid)
    dataset['test']     = torch.utils.data.TensorDataset(X_test, y_test)

    batch_size_list = [2 ** n for n in range(11)] # 1から2^10=1024まで

    output_path = '../../materials/chap08_77_cpu.txt'
    with open(output_path, 'w') as f:
        for batch_size in batch_size_list:
            training_time = MesureTimeWithCPU(dataset=dataset['train'], batch_size=batch_size)

            result = '{}\t{}'.format(batch_size, training_time)
            print('batch_size: ' + result)
            f.write(result + '\n')


"""
batch_size: 1   2.82676100730896
batch_size: 2   1.4694643020629883
batch_size: 4   0.7686736583709717
batch_size: 8   0.4126276969909668
batch_size: 16  0.23431158065795898
batch_size: 32  0.14359354972839355
batch_size: 64  0.09625983238220215
batch_size: 128 0.07216215133666992
batch_size: 256 0.08091354370117188
batch_size: 512 0.054904937744140625
batch_size: 1024        0.05141878128051758
"""
    
