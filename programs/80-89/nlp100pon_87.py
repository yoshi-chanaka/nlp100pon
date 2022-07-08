from nlp100pon_82 import LoadTensorDatasetForRNN
from nlp100pon_84 import CreateW2VEmbedding
from nlp100pon_86 import CNN
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

if __name__ == "__main__":

    dataset, vocab, label_encoder = LoadTensorDatasetForRNN()
    # 単語埋め込み行列の作成
    w2vVocab = [word for word, idx in label_encoder.items() if idx > 0]
    emb_weights = CreateW2VEmbedding(w2vVocab)
    
    batch_size = 64
    train_dataloader    = DataLoader(dataset['train'], batch_size=batch_size, shuffle=True)
    valid_dataloader    = DataLoader(dataset['valid'], batch_size=batch_size, shuffle=False)
    test_dataloader     = DataLoader(dataset['test'], batch_size=batch_size, shuffle=False)
    train_size, valid_size, test_size = len(dataset['train']), len(dataset['valid']), len(dataset['test'])
    print('data size: {}, {}, {}'.format(train_size, valid_size, test_size))
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} device'.format(device))

    model = CNN(
        vocab_size=len(vocab), 
        emb_dim=300, 
        out_channels=128, 
        out_dim=4,
        emb_weights=emb_weights,
    ).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.5)
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
epoch: 1 Train [loss: 1.0416, accuracy: 0.7136], Valid [loss: 0.9821, accuracy: 0.7781]
epoch: 2 Train [loss: 0.9601, accuracy: 0.7879], Valid [loss: 0.9594, accuracy: 0.7931]
epoch: 3 Train [loss: 0.9486, accuracy: 0.7963], Valid [loss: 0.9566, accuracy: 0.7954]
epoch: 4 Train [loss: 0.9382, accuracy: 0.8022], Valid [loss: 0.9508, accuracy: 0.8028]
epoch: 5 Train [loss: 0.9173, accuracy: 0.8337], Valid [loss: 0.9260, accuracy: 0.8313]
epoch: 6 Train [loss: 0.8900, accuracy: 0.8615], Valid [loss: 0.9080, accuracy: 0.8471]
epoch: 7 Train [loss: 0.8621, accuracy: 0.8970], Valid [loss: 0.8830, accuracy: 0.8868]
epoch: 8 Train [loss: 0.8304, accuracy: 0.9293], Valid [loss: 0.8654, accuracy: 0.8913]
epoch: 9 Train [loss: 0.8137, accuracy: 0.9418], Valid [loss: 0.8577, accuracy: 0.8973]
epoch: 10 Train [loss: 0.8038, accuracy: 0.9513], Valid [loss: 0.8595, accuracy: 0.8906]
epoch: 11 Train [loss: 0.7967, accuracy: 0.9564], Valid [loss: 0.8539, accuracy: 0.8981]
epoch: 12 Train [loss: 0.7906, accuracy: 0.9615], Valid [loss: 0.8517, accuracy: 0.9003]
epoch: 13 Train [loss: 0.7865, accuracy: 0.9645], Valid [loss: 0.8533, accuracy: 0.8958]
epoch: 14 Train [loss: 0.7826, accuracy: 0.9684], Valid [loss: 0.8513, accuracy: 0.9010]
epoch: 15 Train [loss: 0.7794, accuracy: 0.9703], Valid [loss: 0.8475, accuracy: 0.9048]
epoch: 16 Train [loss: 0.7769, accuracy: 0.9724], Valid [loss: 0.8463, accuracy: 0.9055]
epoch: 17 Train [loss: 0.7749, accuracy: 0.9743], Valid [loss: 0.8457, accuracy: 0.9048]
epoch: 18 Train [loss: 0.7732, accuracy: 0.9752], Valid [loss: 0.8453, accuracy: 0.9048]
epoch: 19 Train [loss: 0.7719, accuracy: 0.9764], Valid [loss: 0.8462, accuracy: 0.9040]
epoch: 20 Train [loss: 0.7704, accuracy: 0.9769], Valid [loss: 0.8430, accuracy: 0.9093]
epoch: 21 Train [loss: 0.7692, accuracy: 0.9778], Valid [loss: 0.8432, accuracy: 0.9078]
epoch: 22 Train [loss: 0.7686, accuracy: 0.9784], Valid [loss: 0.8417, accuracy: 0.9100]
epoch: 23 Train [loss: 0.7679, accuracy: 0.9786], Valid [loss: 0.8424, accuracy: 0.9078]
epoch: 24 Train [loss: 0.7673, accuracy: 0.9792], Valid [loss: 0.8414, accuracy: 0.9100]
epoch: 25 Train [loss: 0.7669, accuracy: 0.9793], Valid [loss: 0.8412, accuracy: 0.9130]
epoch: 26 Train [loss: 0.7664, accuracy: 0.9798], Valid [loss: 0.8413, accuracy: 0.9093]
epoch: 27 Train [loss: 0.7661, accuracy: 0.9799], Valid [loss: 0.8412, accuracy: 0.9100]
epoch: 28 Train [loss: 0.7658, accuracy: 0.9800], Valid [loss: 0.8402, accuracy: 0.9100]
epoch: 29 Train [loss: 0.7655, accuracy: 0.9803], Valid [loss: 0.8401, accuracy: 0.9123]
epoch: 30 Train [loss: 0.7653, accuracy: 0.9804], Valid [loss: 0.8399, accuracy: 0.9115]

----------------
Train   [loss: 0.7638, accuracy: 0.9807]
Valid   [loss: 0.8331, accuracy: 0.9115]
Test    [loss: 0.8341, accuracy: 0.9085]
"""
