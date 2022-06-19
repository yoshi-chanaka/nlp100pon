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
size of embedding weights: (8492, 300)
number of unregistered words: 928
data size: 10672, 1334, 1334
Using cuda device
epoch: 1 Train [loss: 1.0445, accuracy: 0.7097], Valid [loss: 0.9751, accuracy: 0.7856]
epoch: 2 Train [loss: 0.9594, accuracy: 0.7880], Valid [loss: 0.9570, accuracy: 0.7969]
epoch: 3 Train [loss: 0.9481, accuracy: 0.7963], Valid [loss: 0.9512, accuracy: 0.7999]
epoch: 4 Train [loss: 0.9376, accuracy: 0.8032], Valid [loss: 0.9451, accuracy: 0.8051]
epoch: 5 Train [loss: 0.9199, accuracy: 0.8294], Valid [loss: 0.9267, accuracy: 0.8306]
epoch: 6 Train [loss: 0.8905, accuracy: 0.8629], Valid [loss: 0.9048, accuracy: 0.8538]
epoch: 7 Train [loss: 0.8642, accuracy: 0.8916], Valid [loss: 0.8854, accuracy: 0.8853]
epoch: 8 Train [loss: 0.8326, accuracy: 0.9278], Valid [loss: 0.8596, accuracy: 0.9078]
epoch: 9 Train [loss: 0.8143, accuracy: 0.9412], Valid [loss: 0.8530, accuracy: 0.9070]
epoch: 10 Train [loss: 0.8039, accuracy: 0.9494], Valid [loss: 0.8501, accuracy: 0.9108]
epoch: 11 Train [loss: 0.7962, accuracy: 0.9567], Valid [loss: 0.8479, accuracy: 0.9108]
epoch: 12 Train [loss: 0.7907, accuracy: 0.9615], Valid [loss: 0.8471, accuracy: 0.9078]
epoch: 13 Train [loss: 0.7862, accuracy: 0.9650], Valid [loss: 0.8466, accuracy: 0.9100]
epoch: 14 Train [loss: 0.7824, accuracy: 0.9680], Valid [loss: 0.8429, accuracy: 0.9115]
epoch: 15 Train [loss: 0.7795, accuracy: 0.9704], Valid [loss: 0.8421, accuracy: 0.9123]
epoch: 16 Train [loss: 0.7768, accuracy: 0.9725], Valid [loss: 0.8426, accuracy: 0.9130]
epoch: 17 Train [loss: 0.7749, accuracy: 0.9733], Valid [loss: 0.8411, accuracy: 0.9123]
epoch: 18 Train [loss: 0.7735, accuracy: 0.9745], Valid [loss: 0.8437, accuracy: 0.9100]
epoch: 19 Train [loss: 0.7722, accuracy: 0.9752], Valid [loss: 0.8419, accuracy: 0.9108]
epoch: 20 Train [loss: 0.7710, accuracy: 0.9765], Valid [loss: 0.8428, accuracy: 0.9085]
epoch: 21 Train [loss: 0.7702, accuracy: 0.9769], Valid [loss: 0.8410, accuracy: 0.9070]
epoch: 22 Train [loss: 0.7695, accuracy: 0.9773], Valid [loss: 0.8413, accuracy: 0.9130]
epoch: 23 Train [loss: 0.7691, accuracy: 0.9773], Valid [loss: 0.8417, accuracy: 0.9093]
epoch: 24 Train [loss: 0.7688, accuracy: 0.9775], Valid [loss: 0.8423, accuracy: 0.9100]
epoch: 25 Train [loss: 0.7682, accuracy: 0.9782], Valid [loss: 0.8413, accuracy: 0.9085]
epoch: 26 Train [loss: 0.7678, accuracy: 0.9782], Valid [loss: 0.8412, accuracy: 0.9063]
epoch: 27 Train [loss: 0.7673, accuracy: 0.9787], Valid [loss: 0.8409, accuracy: 0.9040]
epoch: 28 Train [loss: 0.7669, accuracy: 0.9789], Valid [loss: 0.8403, accuracy: 0.9085]
epoch: 29 Train [loss: 0.7666, accuracy: 0.9792], Valid [loss: 0.8407, accuracy: 0.9070]
epoch: 30 Train [loss: 0.7663, accuracy: 0.9797], Valid [loss: 0.8420, accuracy: 0.9070]

----------------
Train   [loss: 0.7650, accuracy: 0.9796]
Valid   [loss: 0.8350, accuracy: 0.9070]
Test    [loss: 0.8309, accuracy: 0.9100]
"""
