from nlp100pon_82 import LoadTensorDatasetForRNN
from nlp100pon_84 import CreateW2VEmbedding
from nlp100pon_85 import BiRNN
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence

if __name__ == "__main__":

    dataset, vocab, label_encoder = LoadTensorDatasetForRNN()
    w2vVocab = [word for word, idx in label_encoder.items() if idx > 0]
    emb_weights = CreateW2VEmbedding(w2vVocab) # Noneにするとword2vecの単語埋め込みは使わない

    batch_size = 64
    train_dataloader    = DataLoader(dataset['train'], batch_size=batch_size, shuffle=True)
    valid_dataloader    = DataLoader(dataset['valid'], batch_size=batch_size, shuffle=False)
    test_dataloader     = DataLoader(dataset['test'], batch_size=batch_size, shuffle=False)
    train_size, valid_size, test_size = len(dataset['train']), len(dataset['valid']), len(dataset['test'])
    print('data size: {}, {}, {}'.format(train_size, valid_size, test_size))
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} device'.format(device))

    for num_layers in range(1, 4):

        model = BiRNN(
            vocab_size=len(vocab), 
            emb_dim=300, 
            hid_dim=64, 
            out_dim=4,
            emb_weights=emb_weights,
            num_layers=num_layers,
        ).to(device)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.99)
        epochs = 30

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

            # print('epoch: {} Train [loss: {:.4f}, accuracy: {:.4f}], Valid [loss: {:.4f}, accuracy: {:.4f}]'.
            #         format(epoch + 1, train_loss, train_acc, valid_loss, valid_acc))


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
        print('{} layer(s):'.format(num_layers))
        print('Train\t[loss: {:.4f}, accuracy: {:.4f}]'.format(train_loss, train_acc))
        print('Valid\t[loss: {:.4f}, accuracy: {:.4f}]'.format(valid_loss, valid_acc))
        print('Test\t[loss: {:.4f}, accuracy: {:.4f}]'.format(test_loss, test_acc))
    
    print('----------------')

"""
loading word2vec ...
creating embedding matrix ...
size of embedding weights: (8492, 300)
number of unregistered words: 928
data size: 10672, 1334, 1334
Using cuda device

----------------
1 layer(s):
Train   [loss: 0.7701, accuracy: 0.9738]
Valid   [loss: 0.8318, accuracy: 0.9078]
Test    [loss: 0.8380, accuracy: 0.9033]

----------------
2 layer(s):
Train   [loss: 0.7799, accuracy: 0.9638]
Valid   [loss: 0.8449, accuracy: 0.8958]
Test    [loss: 0.8478, accuracy: 0.8958]

----------------
3 layer(s):
Train   [loss: 0.7893, accuracy: 0.9553]
Valid   [loss: 0.8444, accuracy: 0.8981]
Test    [loss: 0.8497, accuracy: 0.8921]
----------------
"""
