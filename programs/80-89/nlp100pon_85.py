from nlp100pon_82 import LoadTensorDatasetForRNN
from nlp100pon_84 import CreateW2VEmbedding
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence

class BiRNN(nn.Module):

    def __init__(self, vocab_size, emb_dim, hid_dim, out_dim, emb_weights=None, num_layers=1):
        super(BiRNN, self).__init__()
        self.emb = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_dim)
        if emb_weights != None:
            self.emb.weight = nn.Parameter(emb_weights)
        self.rnn = nn.LSTM(
            input_size=emb_dim, 
            hidden_size=hid_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.linear = nn.Linear(hid_dim * 2, out_dim)

    def forward(self, x, seq_lengths):
        x = self.emb(x)
        x = pack_padded_sequence(x, seq_lengths.cpu().numpy(), batch_first=True)
        packed_output, (ht, ct) = self.rnn(x)
        h = self.linear(torch.cat([ht[0], ht[1]], dim=1)).squeeze(0)
        return F.softmax(h, dim=1)

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

    model = BiRNN(
        vocab_size=len(vocab), 
        emb_dim=300, 
        hid_dim=64, 
        out_dim=4,
        emb_weights=emb_weights,
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
loading word2vec ...
creating embedding matrix ...
size of embedding weights: (7567, 300)
number of unregistered words: 817
data size: 10672, 1334, 1334
Using cuda device
epoch: 1 Train [loss: 1.0618, accuracy: 0.7031], Valid [loss: 0.9605, accuracy: 0.7901]
epoch: 2 Train [loss: 0.9613, accuracy: 0.7821], Valid [loss: 0.9596, accuracy: 0.7901]
epoch: 3 Train [loss: 0.9517, accuracy: 0.7904], Valid [loss: 0.9521, accuracy: 0.7954]
epoch: 4 Train [loss: 0.9343, accuracy: 0.8058], Valid [loss: 0.9289, accuracy: 0.8171]
epoch: 5 Train [loss: 0.9059, accuracy: 0.8367], Valid [loss: 0.9269, accuracy: 0.8253]
epoch: 6 Train [loss: 0.8856, accuracy: 0.8595], Valid [loss: 0.8885, accuracy: 0.8703]
epoch: 7 Train [loss: 0.8558, accuracy: 0.8935], Valid [loss: 0.8797, accuracy: 0.8711]
epoch: 8 Train [loss: 0.8316, accuracy: 0.9151], Valid [loss: 0.8988, accuracy: 0.8486]
epoch: 9 Train [loss: 0.8215, accuracy: 0.9255], Valid [loss: 0.8485, accuracy: 0.9033]
epoch: 10 Train [loss: 0.8096, accuracy: 0.9373], Valid [loss: 0.8494, accuracy: 0.9025]
epoch: 11 Train [loss: 0.8029, accuracy: 0.9438], Valid [loss: 0.8459, accuracy: 0.9048]
epoch: 12 Train [loss: 0.7972, accuracy: 0.9495], Valid [loss: 0.8461, accuracy: 0.9070]
epoch: 13 Train [loss: 0.7925, accuracy: 0.9541], Valid [loss: 0.8464, accuracy: 0.9010]
epoch: 14 Train [loss: 0.7888, accuracy: 0.9575], Valid [loss: 0.8465, accuracy: 0.9025]
epoch: 15 Train [loss: 0.7847, accuracy: 0.9617], Valid [loss: 0.8427, accuracy: 0.9078]
epoch: 16 Train [loss: 0.7823, accuracy: 0.9636], Valid [loss: 0.8428, accuracy: 0.9040]
epoch: 17 Train [loss: 0.7804, accuracy: 0.9656], Valid [loss: 0.8450, accuracy: 0.9040]
epoch: 18 Train [loss: 0.7797, accuracy: 0.9662], Valid [loss: 0.8423, accuracy: 0.9093]
epoch: 19 Train [loss: 0.7801, accuracy: 0.9657], Valid [loss: 0.8470, accuracy: 0.9018]
epoch: 20 Train [loss: 0.7778, accuracy: 0.9680], Valid [loss: 0.8394, accuracy: 0.9123]
epoch: 21 Train [loss: 0.7753, accuracy: 0.9702], Valid [loss: 0.8440, accuracy: 0.9055]
epoch: 22 Train [loss: 0.7746, accuracy: 0.9707], Valid [loss: 0.8441, accuracy: 0.9048]
epoch: 23 Train [loss: 0.7744, accuracy: 0.9711], Valid [loss: 0.8450, accuracy: 0.9040]
epoch: 24 Train [loss: 0.7735, accuracy: 0.9719], Valid [loss: 0.8440, accuracy: 0.9040]
epoch: 25 Train [loss: 0.7731, accuracy: 0.9722], Valid [loss: 0.8444, accuracy: 0.9055]
epoch: 26 Train [loss: 0.7733, accuracy: 0.9721], Valid [loss: 0.8430, accuracy: 0.9048]
epoch: 27 Train [loss: 0.7723, accuracy: 0.9727], Valid [loss: 0.8437, accuracy: 0.9085]
epoch: 28 Train [loss: 0.7720, accuracy: 0.9729], Valid [loss: 0.8423, accuracy: 0.9070]
epoch: 29 Train [loss: 0.7720, accuracy: 0.9729], Valid [loss: 0.8430, accuracy: 0.9055]
epoch: 30 Train [loss: 0.7715, accuracy: 0.9733], Valid [loss: 0.8441, accuracy: 0.9070]

----------------
Train   [loss: 0.7703, accuracy: 0.9734]
Valid   [loss: 0.8372, accuracy: 0.9070]
Test    [loss: 0.8377, accuracy: 0.9033]
"""
