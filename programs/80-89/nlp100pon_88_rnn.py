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
import optuna

def objective(trial):
    
    global train_dataloader, valid_dataloader

    epochs = 20
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    emb_dim = int(trial.suggest_discrete_uniform('emb_dim', 32, 1024, 32))
    hid_dim = int(trial.suggest_discrete_uniform('hid_dim', 16, 256, 16))
    num_layers = int(trial.suggest_discrete_uniform('num_layers', 1, 4, 1))

    model = BiRNN(
        vocab_size=len(vocab), 
        emb_dim=emb_dim, 
        hid_dim=hid_dim, 
        out_dim=4,
        emb_weights=emb_weights,
        num_layers=num_layers,
    ).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.9)

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

    return valid_loss

if __name__ == "__main__":

    dataset, vocab, label_encoder = LoadTensorDatasetForRNN()
    w2vVocab = [word for word, idx in label_encoder.items() if idx > 0]
    emb_weights = None # CreateW2VEmbedding(w2vVocab) # Noneにするとword2vecの単語埋め込みは使わない

    batch_size = 64
    train_dataloader    = DataLoader(dataset['train'], batch_size=batch_size, shuffle=True)
    valid_dataloader    = DataLoader(dataset['valid'], batch_size=batch_size, shuffle=False)
    test_dataloader     = DataLoader(dataset['test'], batch_size=batch_size, shuffle=False)
    train_size, valid_size, test_size = len(dataset['train']), len(dataset['valid']), len(dataset['test'])
    print('data size: {}, {}, {}'.format(train_size, valid_size, test_size))
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} device'.format(device))

    TRIAL_SIZE = 30
    study = optuna.create_study()
    study.optimize(objective, n_trials=TRIAL_SIZE)

    print('best parameters: {}'.format(study.best_params), end='\n\n')

    model = BiRNN(
        vocab_size=len(vocab), 
        emb_dim=int(study.best_params['emb_dim']), 
        hid_dim=int(study.best_params['hid_dim']), 
        out_dim=4,
        emb_weights=emb_weights,
        num_layers=int(study.best_params['num_layers']),
    ).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.9)
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
data size: 10672, 1334, 1334
Using cuda device
[I 2022-06-22 12:41:51,846] A new study created in memory with name: no-name-2a46b65d-38d6-471b-9485-dedacdd7cd9d
[I 2022-06-22 12:42:37,546] Trial 0 finished with value: 0.9533774992157852 and parameters: {'emb_dim': 288.0, 'hid_dim': 144.0, 'num_layers': 4.0}. Best is trial 0 with value: 0.9533774992157852.
[I 2022-06-22 12:42:58,352] Trial 1 finished with value: 0.8868675117549868 and parameters: {'emb_dim': 608.0, 'hid_dim': 64.0, 'num_layers': 1.0}. Best is trial 1 with value: 0.8868675117549868.
[I 2022-06-22 12:43:34,881] Trial 2 finished with value: 0.9528284087173943 and parameters: {'emb_dim': 384.0, 'hid_dim': 16.0, 'num_layers': 3.0}. Best is trial 1 with value: 0.8868675117549868.
[I 2022-06-22 12:44:20,104] Trial 3 finished with value: 0.9634259065230568 and parameters: {'emb_dim': 576.0, 'hid_dim': 96.0, 'num_layers': 4.0}. Best is trial 1 with value: 0.8868675117549868.
[I 2022-06-22 12:44:47,084] Trial 4 finished with value: 0.9181098480453377 and parameters: {'emb_dim': 160.0, 'hid_dim': 80.0, 'num_layers': 2.0}. Best is trial 1 with value: 0.8868675117549868.
[I 2022-06-22 12:45:23,676] Trial 5 finished with value: 0.9424299757698665 and parameters: {'emb_dim': 160.0, 'hid_dim': 32.0, 'num_layers': 3.0}. Best is trial 1 with value: 0.8868675117549868.
[I 2022-06-22 12:45:46,859] Trial 6 finished with value: 1.0016970083988768 and parameters: {'emb_dim': 992.0, 'hid_dim': 16.0, 'num_layers': 1.0}. Best is trial 1 with value: 0.8868675117549868.
[I 2022-06-22 12:46:33,426] Trial 7 finished with value: 0.9403469737680599 and parameters: {'emb_dim': 768.0, 'hid_dim': 112.0, 'num_layers': 4.0}. Best is trial 1 with value: 0.8868675117549868.
[I 2022-06-22 12:47:11,457] Trial 8 finished with value: 0.9101451690765335 and parameters: {'emb_dim': 352.0, 'hid_dim': 208.0, 'num_layers': 3.0}. Best is trial 1 with value: 0.8868675117549868.
[I 2022-06-22 12:48:01,130] Trial 9 finished with value: 0.9372788133292362 and parameters: {'emb_dim': 384.0, 'hid_dim': 224.0, 'num_layers': 4.0}. Best is trial 1 with value: 0.8868675117549868.
[I 2022-06-22 12:48:23,085] Trial 10 finished with value: 0.8789277427021353 and parameters: {'emb_dim': 672.0, 'hid_dim': 160.0, 'num_layers': 1.0}. Best is trial 10 with value: 0.8789277427021353.
[I 2022-06-22 12:48:45,819] Trial 11 finished with value: 0.8836737293889676 and parameters: {'emb_dim': 672.0, 'hid_dim': 176.0, 'num_layers': 1.0}. Best is trial 10 with value: 0.8789277427021353.
[I 2022-06-22 12:49:08,490] Trial 12 finished with value: 0.8709894549185369 and parameters: {'emb_dim': 768.0, 'hid_dim': 176.0, 'num_layers': 1.0}. Best is trial 12 with value: 0.8709894549185369.
[I 2022-06-22 12:49:39,690] Trial 13 finished with value: 0.8622902751505107 and parameters: {'emb_dim': 896.0, 'hid_dim': 160.0, 'num_layers': 2.0}. Best is trial 13 with value: 0.8622902751505107.
[I 2022-06-22 12:50:15,400] Trial 14 finished with value: 0.881595431417897 and parameters: {'emb_dim': 928.0, 'hid_dim': 256.0, 'num_layers': 2.0}. Best is trial 13 with value: 0.8622902751505107.
[I 2022-06-22 12:50:46,208] Trial 15 finished with value: 0.8733180159035473 and parameters: {'emb_dim': 832.0, 'hid_dim': 192.0, 'num_layers': 2.0}. Best is trial 13 with value: 0.8622902751505107.
[I 2022-06-22 12:51:17,063] Trial 16 finished with value: 0.8667931478062848 and parameters: {'emb_dim': 832.0, 'hid_dim': 128.0, 'num_layers': 2.0}. Best is trial 13 with value: 0.8622902751505107.
[I 2022-06-22 12:51:50,020] Trial 17 finished with value: 0.8697140963896104 and parameters: {'emb_dim': 1024.0, 'hid_dim': 128.0, 'num_layers': 2.0}. Best is trial 13 with value: 0.8622902751505107.
[I 2022-06-22 12:52:21,358] Trial 18 finished with value: 0.875188184106189 and parameters: {'emb_dim': 864.0, 'hid_dim': 128.0, 'num_layers': 2.0}. Best is trial 13 with value: 0.8622902751505107.
[I 2022-06-22 12:53:03,169] Trial 19 finished with value: 0.9022920542749865 and parameters: {'emb_dim': 480.0, 'hid_dim': 240.0, 'num_layers': 3.0}. Best is trial 13 with value: 0.8622902751505107.
[I 2022-06-22 12:53:35,405] Trial 20 finished with value: 0.8881851722454203 and parameters: {'emb_dim': 928.0, 'hid_dim': 64.0, 'num_layers': 2.0}. Best is trial 13 with value: 0.8622902751505107.
[I 2022-06-22 12:54:08,822] Trial 21 finished with value: 0.8668185209763283 and parameters: {'emb_dim': 1024.0, 'hid_dim': 128.0, 'num_layers': 2.0}. Best is trial 13 with value: 0.8622902751505107.
[I 2022-06-22 12:54:41,185] Trial 22 finished with value: 0.863011761941295 and parameters: {'emb_dim': 1024.0, 'hid_dim': 144.0, 'num_layers': 2.0}. Best is trial 13 with value: 0.8622902751505107.
[I 2022-06-22 12:55:13,443] Trial 23 finished with value: 0.8694372563169099 and parameters: {'emb_dim': 864.0, 'hid_dim': 160.0, 'num_layers': 2.0}. Best is trial 13 with value: 0.8622902751505107.
[I 2022-06-22 12:55:50,392] Trial 24 finished with value: 0.9057898285506905 and parameters: {'emb_dim': 736.0, 'hid_dim': 96.0, 'num_layers': 3.0}. Best is trial 13 with value: 0.8622902751505107.
[I 2022-06-22 12:56:22,685] Trial 25 finished with value: 0.8688038592931927 and parameters: {'emb_dim': 928.0, 'hid_dim': 144.0, 'num_layers': 2.0}. Best is trial 13 with value: 0.8622902751505107.
[I 2022-06-22 12:57:00,093] Trial 26 finished with value: 0.9924201879544237 and parameters: {'emb_dim': 32.0, 'hid_dim': 192.0, 'num_layers': 3.0}. Best is trial 13 with value: 0.8622902751505107.
[I 2022-06-22 12:57:32,865] Trial 27 finished with value: 0.8672239540935099 and parameters: {'emb_dim': 960.0, 'hid_dim': 160.0, 'num_layers': 2.0}. Best is trial 13 with value: 0.8622902751505107.
[I 2022-06-22 12:57:55,958] Trial 28 finished with value: 0.8845045584431295 and parameters: {'emb_dim': 864.0, 'hid_dim': 112.0, 'num_layers': 1.0}. Best is trial 13 with value: 0.8622902751505107.
[I 2022-06-22 12:58:24,848] Trial 29 finished with value: 0.8878399850367785 and parameters: {'emb_dim': 512.0, 'hid_dim': 144.0, 'num_layers': 2.0}. Best is trial 13 with value: 0.8622902751505107.
best parameters: {'emb_dim': 896.0, 'hid_dim': 160.0, 'num_layers': 2.0}

epoch: 1 Train [loss: 1.0854, accuracy: 0.6793], Valid [loss: 0.9801, accuracy: 0.7721]
epoch: 2 Train [loss: 0.9642, accuracy: 0.7786], Valid [loss: 0.9646, accuracy: 0.7826]
epoch: 3 Train [loss: 0.9434, accuracy: 0.7992], Valid [loss: 0.9562, accuracy: 0.7916]
epoch: 4 Train [loss: 0.9305, accuracy: 0.8084], Valid [loss: 0.9509, accuracy: 0.7969]
epoch: 5 Train [loss: 0.8996, accuracy: 0.8441], Valid [loss: 0.9374, accuracy: 0.8148]
epoch: 6 Train [loss: 0.8659, accuracy: 0.8800], Valid [loss: 0.9080, accuracy: 0.8426]
epoch: 7 Train [loss: 0.8484, accuracy: 0.8949], Valid [loss: 0.9081, accuracy: 0.8456]
epoch: 8 Train [loss: 0.8361, accuracy: 0.9045], Valid [loss: 0.9035, accuracy: 0.8441]
epoch: 9 Train [loss: 0.8111, accuracy: 0.9387], Valid [loss: 0.8869, accuracy: 0.8591]
epoch: 10 Train [loss: 0.7890, accuracy: 0.9591], Valid [loss: 0.8862, accuracy: 0.8621]
epoch: 11 Train [loss: 0.7803, accuracy: 0.9667], Valid [loss: 0.8794, accuracy: 0.8711]
epoch: 12 Train [loss: 0.7729, accuracy: 0.9728], Valid [loss: 0.8728, accuracy: 0.8763]
epoch: 13 Train [loss: 0.7701, accuracy: 0.9753], Valid [loss: 0.8754, accuracy: 0.8711]
epoch: 14 Train [loss: 0.7664, accuracy: 0.9790], Valid [loss: 0.8744, accuracy: 0.8748]
epoch: 15 Train [loss: 0.7655, accuracy: 0.9799], Valid [loss: 0.8731, accuracy: 0.8748]
epoch: 16 Train [loss: 0.7634, accuracy: 0.9816], Valid [loss: 0.8765, accuracy: 0.8696]
epoch: 17 Train [loss: 0.7616, accuracy: 0.9833], Valid [loss: 0.8715, accuracy: 0.8763]
epoch: 18 Train [loss: 0.7609, accuracy: 0.9840], Valid [loss: 0.8704, accuracy: 0.8778]
epoch: 19 Train [loss: 0.7609, accuracy: 0.9839], Valid [loss: 0.8726, accuracy: 0.8763]
epoch: 20 Train [loss: 0.7608, accuracy: 0.9841], Valid [loss: 0.8742, accuracy: 0.8726]
epoch: 21 Train [loss: 0.7606, accuracy: 0.9843], Valid [loss: 0.8707, accuracy: 0.8778]
epoch: 22 Train [loss: 0.7602, accuracy: 0.9846], Valid [loss: 0.8709, accuracy: 0.8786]
epoch: 23 Train [loss: 0.7601, accuracy: 0.9847], Valid [loss: 0.8700, accuracy: 0.8808]
epoch: 24 Train [loss: 0.7599, accuracy: 0.9849], Valid [loss: 0.8694, accuracy: 0.8823]
epoch: 25 Train [loss: 0.7598, accuracy: 0.9850], Valid [loss: 0.8686, accuracy: 0.8816]
epoch: 26 Train [loss: 0.7596, accuracy: 0.9852], Valid [loss: 0.8680, accuracy: 0.8786]
epoch: 27 Train [loss: 0.7596, accuracy: 0.9852], Valid [loss: 0.8715, accuracy: 0.8793]
epoch: 28 Train [loss: 0.7594, accuracy: 0.9854], Valid [loss: 0.8701, accuracy: 0.8793]
epoch: 29 Train [loss: 0.7593, accuracy: 0.9856], Valid [loss: 0.8703, accuracy: 0.8756]
epoch: 30 Train [loss: 0.7593, accuracy: 0.9855], Valid [loss: 0.8699, accuracy: 0.8756]

----------------
Train   [loss: 0.7580, accuracy: 0.9857]
Valid   [loss: 0.8626, accuracy: 0.8756]
Test    [loss: 0.8606, accuracy: 0.8808]
"""
