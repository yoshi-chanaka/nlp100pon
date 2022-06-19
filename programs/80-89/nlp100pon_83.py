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
epoch: 1 Train [loss: 1.1644, accuracy: 0.6002], Valid [loss: 1.0289, accuracy: 0.7286]
epoch: 2 Train [loss: 0.9942, accuracy: 0.7523], Valid [loss: 1.0047, accuracy: 0.7444]
epoch: 3 Train [loss: 0.9605, accuracy: 0.7836], Valid [loss: 0.9878, accuracy: 0.7571]
epoch: 4 Train [loss: 0.9461, accuracy: 0.7968], Valid [loss: 0.9695, accuracy: 0.7796]
epoch: 5 Train [loss: 0.9380, accuracy: 0.8042], Valid [loss: 0.9721, accuracy: 0.7766]
epoch: 6 Train [loss: 0.9327, accuracy: 0.8083], Valid [loss: 0.9707, accuracy: 0.7774]
epoch: 7 Train [loss: 0.9279, accuracy: 0.8104], Valid [loss: 0.9661, accuracy: 0.7826]
epoch: 8 Train [loss: 0.9177, accuracy: 0.8153], Valid [loss: 0.9656, accuracy: 0.7826]
epoch: 9 Train [loss: 0.9004, accuracy: 0.8477], Valid [loss: 0.9602, accuracy: 0.7879]
epoch: 10 Train [loss: 0.8781, accuracy: 0.8731], Valid [loss: 0.9581, accuracy: 0.7871]
epoch: 11 Train [loss: 0.8594, accuracy: 0.8907], Valid [loss: 0.9523, accuracy: 0.7894]
epoch: 12 Train [loss: 0.8399, accuracy: 0.9102], Valid [loss: 0.9322, accuracy: 0.8186]
epoch: 13 Train [loss: 0.8234, accuracy: 0.9286], Valid [loss: 0.9277, accuracy: 0.8193]
epoch: 14 Train [loss: 0.8064, accuracy: 0.9457], Valid [loss: 0.9325, accuracy: 0.8073]
epoch: 15 Train [loss: 0.7928, accuracy: 0.9587], Valid [loss: 0.9346, accuracy: 0.8126]
epoch: 16 Train [loss: 0.7847, accuracy: 0.9653], Valid [loss: 0.9258, accuracy: 0.8216]
epoch: 17 Train [loss: 0.7792, accuracy: 0.9695], Valid [loss: 0.9235, accuracy: 0.8231]
epoch: 18 Train [loss: 0.7757, accuracy: 0.9726], Valid [loss: 0.9279, accuracy: 0.8208]
epoch: 19 Train [loss: 0.7729, accuracy: 0.9751], Valid [loss: 0.9193, accuracy: 0.8291]
epoch: 20 Train [loss: 0.7706, accuracy: 0.9769], Valid [loss: 0.9192, accuracy: 0.8283]
epoch: 21 Train [loss: 0.7682, accuracy: 0.9786], Valid [loss: 0.9181, accuracy: 0.8276]
epoch: 22 Train [loss: 0.7660, accuracy: 0.9802], Valid [loss: 0.9146, accuracy: 0.8336]
epoch: 23 Train [loss: 0.7655, accuracy: 0.9808], Valid [loss: 0.9156, accuracy: 0.8321]
epoch: 24 Train [loss: 0.7650, accuracy: 0.9809], Valid [loss: 0.9123, accuracy: 0.8373]
epoch: 25 Train [loss: 0.7633, accuracy: 0.9820], Valid [loss: 0.9107, accuracy: 0.8381]
epoch: 26 Train [loss: 0.7627, accuracy: 0.9827], Valid [loss: 0.9105, accuracy: 0.8388]
epoch: 27 Train [loss: 0.7621, accuracy: 0.9831], Valid [loss: 0.9086, accuracy: 0.8411]
epoch: 28 Train [loss: 0.7615, accuracy: 0.9836], Valid [loss: 0.9074, accuracy: 0.8418]
epoch: 29 Train [loss: 0.7612, accuracy: 0.9838], Valid [loss: 0.9061, accuracy: 0.8471]
epoch: 30 Train [loss: 0.7611, accuracy: 0.9839], Valid [loss: 0.9054, accuracy: 0.8486]
epoch: 31 Train [loss: 0.7610, accuracy: 0.9840], Valid [loss: 0.9049, accuracy: 0.8478]
epoch: 32 Train [loss: 0.7611, accuracy: 0.9840], Valid [loss: 0.9081, accuracy: 0.8418]
epoch: 33 Train [loss: 0.7609, accuracy: 0.9841], Valid [loss: 0.9060, accuracy: 0.8441]
epoch: 34 Train [loss: 0.7609, accuracy: 0.9839], Valid [loss: 0.9081, accuracy: 0.8381]
epoch: 35 Train [loss: 0.7657, accuracy: 0.9813], Valid [loss: 0.9135, accuracy: 0.8343]
epoch: 36 Train [loss: 0.7644, accuracy: 0.9818], Valid [loss: 0.9102, accuracy: 0.8358]
epoch: 37 Train [loss: 0.7627, accuracy: 0.9829], Valid [loss: 0.9105, accuracy: 0.8351]
epoch: 38 Train [loss: 0.7628, accuracy: 0.9832], Valid [loss: 0.9082, accuracy: 0.8396]
epoch: 39 Train [loss: 0.7635, accuracy: 0.9825], Valid [loss: 0.9009, accuracy: 0.8478]
epoch: 40 Train [loss: 0.7612, accuracy: 0.9845], Valid [loss: 0.9045, accuracy: 0.8441]
epoch: 41 Train [loss: 0.7599, accuracy: 0.9856], Valid [loss: 0.9074, accuracy: 0.8418]
epoch: 42 Train [loss: 0.7592, accuracy: 0.9859], Valid [loss: 0.9044, accuracy: 0.8418]
epoch: 43 Train [loss: 0.7589, accuracy: 0.9861], Valid [loss: 0.9041, accuracy: 0.8426]
epoch: 44 Train [loss: 0.7587, accuracy: 0.9862], Valid [loss: 0.9040, accuracy: 0.8418]
epoch: 45 Train [loss: 0.7586, accuracy: 0.9862], Valid [loss: 0.9032, accuracy: 0.8426]
epoch: 46 Train [loss: 0.7585, accuracy: 0.9863], Valid [loss: 0.9013, accuracy: 0.8478]
epoch: 47 Train [loss: 0.7584, accuracy: 0.9864], Valid [loss: 0.9011, accuracy: 0.8456]
epoch: 48 Train [loss: 0.7583, accuracy: 0.9865], Valid [loss: 0.9029, accuracy: 0.8456]
epoch: 49 Train [loss: 0.7582, accuracy: 0.9866], Valid [loss: 0.9014, accuracy: 0.8463]
epoch: 50 Train [loss: 0.7581, accuracy: 0.9867], Valid [loss: 0.9012, accuracy: 0.8463]

----------------
Train   [loss: 0.7568, accuracy: 0.9868]
Valid   [loss: 0.8941, accuracy: 0.8463]
Test    [loss: 0.8797, accuracy: 0.8591]
"""
