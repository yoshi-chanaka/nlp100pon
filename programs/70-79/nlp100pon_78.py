from nlp100pon_70 import LoadData
from nlp100pon_71 import MLP
from nlp100pon_77 import MesureTimeWithCPU
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


def MesureTimeWithGPU(dataset, batch_size, epochs=1):

    device = 'cuda'
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    size = len(dataloader.dataset)

    model = MLP(in_dim=300, out_dim=4).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.99)

    training_time = 0
    for epoch in range(epochs):
        
        model.train()
        for X, y in dataloader:

            X, y = X.to(device), y.to(device)
            torch.cuda.synchronize()
            start = time.time()
            
            pred = model(X)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            torch.cuda.synchronize()
            training_time += time.time() - start
    
    return training_time


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

    output_path = '../../materials/chap08_78.txt'
    print('\tcpu\tgpu\n------------------------')
    cpu_training_time_list, gpu_training_time_list = [], []
    with open(output_path, 'w') as f:
        for batch_size in batch_size_list:
            training_time_cpu = MesureTimeWithCPU(dataset=dataset['train'], batch_size=batch_size)
            training_time_gpu = MesureTimeWithGPU(dataset=dataset['train'], batch_size=batch_size)
            cpu_training_time_list.append(training_time_cpu)
            gpu_training_time_list.append(training_time_gpu)

            result = '{}\t{}\t{}'.format(batch_size, training_time_cpu, training_time_gpu)
            print('batch_size: ' + result)
            f.write(result + '\n')
    
    fig, ax = plt.subplots()

    ax.plot(batch_size_list, cpu_training_time_list, label='CPU')
    ax.plot(batch_size_list, gpu_training_time_list, label='GPU')
    ax.set_xscale('log', basex=2)
    ax.set_yscale('log', basey=10)
    ax.set_xlabel('batch_size')
    ax.set_ylabel('time')
    ax.set_xticks(batch_size_list)
    plt.legend()
    plt.savefig('../../figures/chap08_78_measuretime.jpg')

"""
        cpu     gpu
------------------------
batch_size: 1   2.794757127761841       6.890453815460205
batch_size: 2   1.4904403686523438      2.634692430496216
batch_size: 4   0.7940518856048584      1.6157283782958984
batch_size: 8   0.43380236625671387     0.7585523128509521
batch_size: 16  0.2533540725708008      0.32651495933532715
batch_size: 32  0.14553570747375488     0.16519665718078613
batch_size: 64  0.09857439994812012     0.07264351844787598
batch_size: 128 0.07480311393737793     0.0363001823425293
batch_size: 256 0.08369755744934082     0.018490076065063477
batch_size: 512 0.058298349380493164    0.009848356246948242
batch_size: 1024        0.05446743965148926     0.006695747375488281
"""
    
