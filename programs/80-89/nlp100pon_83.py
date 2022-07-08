from nlp100pon_80 import ConvertDatasetToIDSeqSet, LoadData
from nlp100pon_81 import RNN
from nlp100pon_82 import LoadTensorDatasetForRNN
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence


if __name__ == "__main__":
    
    dataset, vocab, _ = LoadTensorDatasetForRNN()

    batch_size = 64
    train_dataloader    = DataLoader(dataset['train'], batch_size=batch_size, shuffle=True)
    valid_dataloader    = DataLoader(dataset['valid'], batch_size=batch_size, shuffle=False)
    test_dataloader     = DataLoader(dataset['test'], batch_size=batch_size, shuffle=False)
    train_size, valid_size, test_size = len(dataset['train']), len(dataset['valid']), len(dataset['test'])
    print('data size: {}, {}, {}'.format(train_size, valid_size, test_size))
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} device'.format(device))

    model = RNN(
        vocab_size=len(vocab), 
        emb_dim=300, 
        hid_dim=64, 
        out_dim=4
    ).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.99)
    epochs = 50

    for epoch in range(epochs):
        
        model.train()
        train_loss, train_num_correct = 0, 0
        for X, seq_lengths, y in train_dataloader:
            
            seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)

            X, y = X[perm_idx].to(device), y[perm_idx].to(device)
            pred = model(X, seq_lengths)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_size
            train_num_correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        
        train_loss /= train_size
        train_acc = train_num_correct / train_size

        model.eval()
        valid_loss, valid_num_correct = 0, 0
        for X, seq_lengths, y in valid_dataloader:

            seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)

            X, y = X[perm_idx].to(device), y[perm_idx].to(device)
            pred = model(X, seq_lengths)
            valid_loss += loss_fn(pred, y).item() * batch_size
            valid_num_correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        
        valid_loss /= valid_size
        valid_acc = valid_num_correct / valid_size

        print('epoch: {} Train [loss: {:.4f}, accuracy: {:.4f}], Valid [loss: {:.4f}, accuracy: {:.4f}]'.
                format(epoch + 1, train_loss, train_acc, valid_loss, valid_acc))


    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    
    k = 'train'
    X, seq_lengths, y = dataset[k][:]
    seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
    X, y = X[perm_idx].to(device), y[perm_idx]
    pred = model.forward(X, seq_lengths).cpu()
    train_loss = loss_fn(pred, y)
    train_acc = (pred.argmax(1) == y).type(torch.float).sum().item() / train_size

    k = 'valid'
    X, seq_lengths, y = dataset[k][:]
    seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
    X, y = X[perm_idx].to(device), y[perm_idx]
    pred = model.forward(X, seq_lengths).cpu()
    valid_loss = loss_fn(pred, y)
    valid_acc = (pred.argmax(1) == y).type(torch.float).sum().item() / valid_size

    k = 'test'
    X, seq_lengths, y = dataset[k][:]
    seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
    X, y = X[perm_idx].to(device), y[perm_idx]
    pred = model.forward(X, seq_lengths).cpu()
    test_loss = loss_fn(pred, y)
    test_acc = (pred.argmax(1) == y).type(torch.float).sum().item() / test_size

    print('\n----------------')
    print('Train\t[loss: {:.4f}, accuracy: {:.4f}]'.format(train_loss, train_acc))
    print('Valid\t[loss: {:.4f}, accuracy: {:.4f}]'.format(valid_loss, valid_acc))
    print('Test\t[loss: {:.4f}, accuracy: {:.4f}]'.format(test_loss, test_acc))

"""
data size: 10672, 1334, 1334
Using cuda device
epoch: 1 Train [loss: 1.1643, accuracy: 0.6059], Valid [loss: 1.0400, accuracy: 0.7144]
epoch: 2 Train [loss: 0.9932, accuracy: 0.7518], Valid [loss: 0.9917, accuracy: 0.7571]
epoch: 3 Train [loss: 0.9587, accuracy: 0.7855], Valid [loss: 0.9838, accuracy: 0.7616]
epoch: 4 Train [loss: 0.9453, accuracy: 0.7985], Valid [loss: 0.9683, accuracy: 0.7834]
epoch: 5 Train [loss: 0.9360, accuracy: 0.8056], Valid [loss: 0.9650, accuracy: 0.7849]
epoch: 6 Train [loss: 0.9307, accuracy: 0.8097], Valid [loss: 0.9611, accuracy: 0.7886]
epoch: 7 Train [loss: 0.9257, accuracy: 0.8112], Valid [loss: 0.9582, accuracy: 0.7894]
epoch: 8 Train [loss: 0.9135, accuracy: 0.8209], Valid [loss: 0.9609, accuracy: 0.7856]
epoch: 9 Train [loss: 0.8961, accuracy: 0.8534], Valid [loss: 0.9526, accuracy: 0.7991]
epoch: 10 Train [loss: 0.8708, accuracy: 0.8794], Valid [loss: 0.9395, accuracy: 0.8133]
epoch: 11 Train [loss: 0.8467, accuracy: 0.9063], Valid [loss: 0.9451, accuracy: 0.8066]
epoch: 12 Train [loss: 0.8283, accuracy: 0.9252], Valid [loss: 0.9318, accuracy: 0.8163]
epoch: 13 Train [loss: 0.8111, accuracy: 0.9419], Valid [loss: 0.9310, accuracy: 0.8148]
epoch: 14 Train [loss: 0.7959, accuracy: 0.9567], Valid [loss: 0.9247, accuracy: 0.8223]
epoch: 15 Train [loss: 0.7857, accuracy: 0.9649], Valid [loss: 0.9244, accuracy: 0.8163]
epoch: 16 Train [loss: 0.7790, accuracy: 0.9691], Valid [loss: 0.9225, accuracy: 0.8238]
epoch: 17 Train [loss: 0.7745, accuracy: 0.9737], Valid [loss: 0.9212, accuracy: 0.8231]
epoch: 18 Train [loss: 0.7715, accuracy: 0.9759], Valid [loss: 0.9172, accuracy: 0.8313]
epoch: 19 Train [loss: 0.7691, accuracy: 0.9775], Valid [loss: 0.9128, accuracy: 0.8381]
epoch: 20 Train [loss: 0.7674, accuracy: 0.9783], Valid [loss: 0.9121, accuracy: 0.8373]
epoch: 21 Train [loss: 0.7663, accuracy: 0.9792], Valid [loss: 0.9117, accuracy: 0.8381]
epoch: 22 Train [loss: 0.7654, accuracy: 0.9799], Valid [loss: 0.9109, accuracy: 0.8381]
epoch: 23 Train [loss: 0.7650, accuracy: 0.9803], Valid [loss: 0.9116, accuracy: 0.8358]
epoch: 24 Train [loss: 0.7645, accuracy: 0.9806], Valid [loss: 0.9102, accuracy: 0.8381]
epoch: 25 Train [loss: 0.7641, accuracy: 0.9809], Valid [loss: 0.9094, accuracy: 0.8351]
epoch: 26 Train [loss: 0.7641, accuracy: 0.9810], Valid [loss: 0.9103, accuracy: 0.8358]
epoch: 27 Train [loss: 0.7639, accuracy: 0.9811], Valid [loss: 0.9094, accuracy: 0.8388]
epoch: 28 Train [loss: 0.7635, accuracy: 0.9814], Valid [loss: 0.9063, accuracy: 0.8418]
epoch: 29 Train [loss: 0.7632, accuracy: 0.9816], Valid [loss: 0.9057, accuracy: 0.8411]
epoch: 30 Train [loss: 0.7634, accuracy: 0.9816], Valid [loss: 0.9060, accuracy: 0.8418]
epoch: 31 Train [loss: 0.7630, accuracy: 0.9819], Valid [loss: 0.9059, accuracy: 0.8456]
epoch: 32 Train [loss: 0.7638, accuracy: 0.9815], Valid [loss: 0.9083, accuracy: 0.8381]
epoch: 33 Train [loss: 0.7646, accuracy: 0.9808], Valid [loss: 0.9088, accuracy: 0.8403]
epoch: 34 Train [loss: 0.7637, accuracy: 0.9819], Valid [loss: 0.9071, accuracy: 0.8418]
epoch: 35 Train [loss: 0.7637, accuracy: 0.9819], Valid [loss: 0.9049, accuracy: 0.8418]
epoch: 36 Train [loss: 0.7624, accuracy: 0.9827], Valid [loss: 0.9029, accuracy: 0.8433]
epoch: 37 Train [loss: 0.7617, accuracy: 0.9832], Valid [loss: 0.9033, accuracy: 0.8441]
epoch: 38 Train [loss: 0.7620, accuracy: 0.9829], Valid [loss: 0.9152, accuracy: 0.8321]
epoch: 39 Train [loss: 0.7660, accuracy: 0.9802], Valid [loss: 0.9147, accuracy: 0.8313]
epoch: 40 Train [loss: 0.7639, accuracy: 0.9818], Valid [loss: 0.9142, accuracy: 0.8343]
epoch: 41 Train [loss: 0.7616, accuracy: 0.9837], Valid [loss: 0.9112, accuracy: 0.8336]
epoch: 42 Train [loss: 0.7612, accuracy: 0.9838], Valid [loss: 0.9086, accuracy: 0.8388]
epoch: 43 Train [loss: 0.7608, accuracy: 0.9841], Valid [loss: 0.9091, accuracy: 0.8358]
epoch: 44 Train [loss: 0.7605, accuracy: 0.9843], Valid [loss: 0.9087, accuracy: 0.8358]
epoch: 45 Train [loss: 0.7606, accuracy: 0.9844], Valid [loss: 0.9078, accuracy: 0.8373]
epoch: 46 Train [loss: 0.7604, accuracy: 0.9844], Valid [loss: 0.9071, accuracy: 0.8403]
epoch: 47 Train [loss: 0.7603, accuracy: 0.9844], Valid [loss: 0.9073, accuracy: 0.8396]
epoch: 48 Train [loss: 0.7603, accuracy: 0.9844], Valid [loss: 0.9065, accuracy: 0.8396]
epoch: 49 Train [loss: 0.7603, accuracy: 0.9844], Valid [loss: 0.9068, accuracy: 0.8403]
epoch: 50 Train [loss: 0.7603, accuracy: 0.9844], Valid [loss: 0.9066, accuracy: 0.8403]

----------------
Train   [loss: 0.7591, accuracy: 0.9845]
Valid   [loss: 0.8998, accuracy: 0.8403]
Test    [loss: 0.8840, accuracy: 0.8568]
"""
