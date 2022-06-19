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
        self.linear = nn.Linear(hid_dim, out_dim)

    def forward(self, x, seq_lengths):
        x = self.emb(x)
        x = pack_padded_sequence(x, seq_lengths.cpu().numpy(), batch_first=True)
        packed_output, (ht, ct) = self.rnn(x)
        h = self.linear(ht[-1]).squeeze(0)
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
size of embedding weights: (8492, 300)
number of unregistered words: 928
data size: 10672, 1334, 1334
Using cuda device
epoch: 1 Train [loss: 1.0698, accuracy: 0.6971], Valid [loss: 1.0151, accuracy: 0.7331]
epoch: 2 Train [loss: 0.9618, accuracy: 0.7809], Valid [loss: 0.9534, accuracy: 0.7969]
epoch: 3 Train [loss: 0.9522, accuracy: 0.7894], Valid [loss: 0.9471, accuracy: 0.7946]
epoch: 4 Train [loss: 0.9283, accuracy: 0.8144], Valid [loss: 0.9191, accuracy: 0.8366]
epoch: 5 Train [loss: 0.9039, accuracy: 0.8370], Valid [loss: 0.9059, accuracy: 0.8411]
epoch: 6 Train [loss: 0.8833, accuracy: 0.8599], Valid [loss: 0.8730, accuracy: 0.8846]
epoch: 7 Train [loss: 0.8466, accuracy: 0.9004], Valid [loss: 0.8609, accuracy: 0.8913]
epoch: 8 Train [loss: 0.8262, accuracy: 0.9211], Valid [loss: 0.8571, accuracy: 0.8951]
epoch: 9 Train [loss: 0.8187, accuracy: 0.9273], Valid [loss: 0.8775, accuracy: 0.8673]
epoch: 10 Train [loss: 0.8079, accuracy: 0.9386], Valid [loss: 0.8427, accuracy: 0.9100]
epoch: 11 Train [loss: 0.8058, accuracy: 0.9403], Valid [loss: 0.8501, accuracy: 0.9018]
epoch: 12 Train [loss: 0.7978, accuracy: 0.9483], Valid [loss: 0.8543, accuracy: 0.8966]
epoch: 13 Train [loss: 0.7944, accuracy: 0.9514], Valid [loss: 0.8429, accuracy: 0.9018]
epoch: 14 Train [loss: 0.7887, accuracy: 0.9576], Valid [loss: 0.8332, accuracy: 0.9160]
epoch: 15 Train [loss: 0.7848, accuracy: 0.9619], Valid [loss: 0.8336, accuracy: 0.9175]
epoch: 16 Train [loss: 0.7828, accuracy: 0.9637], Valid [loss: 0.8308, accuracy: 0.9213]
epoch: 17 Train [loss: 0.7803, accuracy: 0.9656], Valid [loss: 0.8500, accuracy: 0.8988]
epoch: 18 Train [loss: 0.7788, accuracy: 0.9674], Valid [loss: 0.8370, accuracy: 0.9123]
epoch: 19 Train [loss: 0.7783, accuracy: 0.9669], Valid [loss: 0.8379, accuracy: 0.9115]
epoch: 20 Train [loss: 0.7772, accuracy: 0.9686], Valid [loss: 0.8411, accuracy: 0.9055]
epoch: 21 Train [loss: 0.7755, accuracy: 0.9699], Valid [loss: 0.8371, accuracy: 0.9115]
epoch: 22 Train [loss: 0.7748, accuracy: 0.9706], Valid [loss: 0.8360, accuracy: 0.9123]
epoch: 23 Train [loss: 0.7751, accuracy: 0.9701], Valid [loss: 0.8432, accuracy: 0.9025]
epoch: 24 Train [loss: 0.7735, accuracy: 0.9717], Valid [loss: 0.8383, accuracy: 0.9078]
epoch: 25 Train [loss: 0.7733, accuracy: 0.9720], Valid [loss: 0.8363, accuracy: 0.9115]
epoch: 26 Train [loss: 0.7740, accuracy: 0.9710], Valid [loss: 0.8382, accuracy: 0.9100]
epoch: 27 Train [loss: 0.7730, accuracy: 0.9722], Valid [loss: 0.8375, accuracy: 0.9130]
epoch: 28 Train [loss: 0.7732, accuracy: 0.9722], Valid [loss: 0.8351, accuracy: 0.9153]
epoch: 29 Train [loss: 0.7719, accuracy: 0.9731], Valid [loss: 0.8362, accuracy: 0.9138]
epoch: 30 Train [loss: 0.7719, accuracy: 0.9730], Valid [loss: 0.8383, accuracy: 0.9100]

----------------
Train   [loss: 0.7702, accuracy: 0.9735]
Valid   [loss: 0.8316, accuracy: 0.9100]
Test    [loss: 0.8289, accuracy: 0.9138]
"""
