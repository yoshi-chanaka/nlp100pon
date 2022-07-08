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
size of embedding weights: (7567, 300)
number of unregistered words: 817
data size: 10672, 1334, 1334
Using cuda device
epoch: 1 Train [loss: 1.0764, accuracy: 0.6854], Valid [loss: 0.9902, accuracy: 0.7586]
epoch: 2 Train [loss: 0.9635, accuracy: 0.7802], Valid [loss: 0.9679, accuracy: 0.7811]
epoch: 3 Train [loss: 0.9535, accuracy: 0.7901], Valid [loss: 0.9663, accuracy: 0.7826]
epoch: 4 Train [loss: 0.9413, accuracy: 0.7986], Valid [loss: 0.9886, accuracy: 0.7601]
epoch: 5 Train [loss: 0.9189, accuracy: 0.8245], Valid [loss: 0.9118, accuracy: 0.8328]
epoch: 6 Train [loss: 0.8927, accuracy: 0.8508], Valid [loss: 0.9033, accuracy: 0.8448]
epoch: 7 Train [loss: 0.8754, accuracy: 0.8684], Valid [loss: 0.8824, accuracy: 0.8793]
epoch: 8 Train [loss: 0.8444, accuracy: 0.9049], Valid [loss: 0.8773, accuracy: 0.8741]
epoch: 9 Train [loss: 0.8232, accuracy: 0.9249], Valid [loss: 0.8592, accuracy: 0.8951]
epoch: 10 Train [loss: 0.8146, accuracy: 0.9321], Valid [loss: 0.8472, accuracy: 0.9033]
epoch: 11 Train [loss: 0.8082, accuracy: 0.9375], Valid [loss: 0.8542, accuracy: 0.8988]
epoch: 12 Train [loss: 0.7998, accuracy: 0.9465], Valid [loss: 0.8446, accuracy: 0.9070]
epoch: 13 Train [loss: 0.7943, accuracy: 0.9519], Valid [loss: 0.8465, accuracy: 0.9055]
epoch: 14 Train [loss: 0.7928, accuracy: 0.9531], Valid [loss: 0.8477, accuracy: 0.9033]
epoch: 15 Train [loss: 0.7884, accuracy: 0.9577], Valid [loss: 0.8409, accuracy: 0.9085]
epoch: 16 Train [loss: 0.7844, accuracy: 0.9623], Valid [loss: 0.8406, accuracy: 0.9085]
epoch: 17 Train [loss: 0.7846, accuracy: 0.9617], Valid [loss: 0.8452, accuracy: 0.9048]
epoch: 18 Train [loss: 0.7839, accuracy: 0.9619], Valid [loss: 0.8402, accuracy: 0.9070]
epoch: 19 Train [loss: 0.7800, accuracy: 0.9656], Valid [loss: 0.8388, accuracy: 0.9130]
epoch: 20 Train [loss: 0.7774, accuracy: 0.9681], Valid [loss: 0.8366, accuracy: 0.9130]
epoch: 21 Train [loss: 0.7767, accuracy: 0.9684], Valid [loss: 0.8549, accuracy: 0.8921]
epoch: 22 Train [loss: 0.7763, accuracy: 0.9693], Valid [loss: 0.8387, accuracy: 0.9108]
epoch: 23 Train [loss: 0.7754, accuracy: 0.9699], Valid [loss: 0.8341, accuracy: 0.9183]
epoch: 24 Train [loss: 0.7743, accuracy: 0.9709], Valid [loss: 0.8359, accuracy: 0.9138]
epoch: 25 Train [loss: 0.7739, accuracy: 0.9711], Valid [loss: 0.8369, accuracy: 0.9115]
epoch: 26 Train [loss: 0.7738, accuracy: 0.9713], Valid [loss: 0.8360, accuracy: 0.9153]
epoch: 27 Train [loss: 0.7735, accuracy: 0.9714], Valid [loss: 0.8349, accuracy: 0.9153]
epoch: 28 Train [loss: 0.7731, accuracy: 0.9717], Valid [loss: 0.8350, accuracy: 0.9138]
epoch: 29 Train [loss: 0.7734, accuracy: 0.9718], Valid [loss: 0.8476, accuracy: 0.8981]
epoch: 30 Train [loss: 0.7734, accuracy: 0.9714], Valid [loss: 0.8343, accuracy: 0.9160]

----------------
Train   [loss: 0.7715, accuracy: 0.9723]
Valid   [loss: 0.8278, accuracy: 0.9160]
Test    [loss: 0.8394, accuracy: 0.9033]
"""
