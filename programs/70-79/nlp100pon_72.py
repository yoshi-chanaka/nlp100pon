from nlp100pon_70 import LoadData
from nlp100pon_71 import MLP
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


if __name__ == "__main__":

    path = '../../data/NewsAggregatorDataset/chap08_avgembed.pickle'
    X, Y = LoadData(path)    
    X_train, y_train = \
        torch.from_numpy(X['train'].astype(np.float32))[:4], torch.tensor(Y['train'])[:4]

    model = MLP(in_dim=300, out_dim=4)
    pred = model.forward(X_train)
        
    loss = (torch.eye(pred.shape[1])[y_train] * (- torch.log(pred))).sum(axis=1)
    print('loss:\t', loss)

    loss_fn = nn.CrossEntropyLoss(reduction='none')
    loss = loss_fn(pred, y_train)
    print('loss(nn.CrossEntropyLoss):\t', loss)

    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(pred, y_train)
    loss.backward()
    print('\ngradient: ')
    for params in model.parameters():
        print(params.grad)
    
"""
loss:    tensor([1.3210, 1.3503, 1.3180, 1.3963], grad_fn=<SumBackward1>)
loss(nn.CrossEntropyLoss):       tensor([1.3695, 1.3772, 1.3687, 1.3888], grad_fn=<NllLossBackward0>)

gradient:
tensor([[ 2.0373e-03,  2.2041e-03, -9.9229e-04,  ..., -8.1170e-03,
         -1.3147e-04,  5.3827e-03],
        [-6.5682e-04,  2.8339e-03, -3.9469e-05,  ..., -2.1227e-03,
         -2.1325e-03, -2.1228e-03],
        [-6.2865e-04, -7.7657e-03,  1.1397e-03,  ...,  1.2296e-02,
          4.4651e-03, -1.2464e-03],
        [-7.5182e-04,  2.7278e-03, -1.0795e-04,  ..., -2.0567e-03,
         -2.2011e-03, -2.0135e-03]])
"""