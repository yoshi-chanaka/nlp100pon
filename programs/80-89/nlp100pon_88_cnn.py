from nlp100pon_82 import LoadTensorDatasetForRNN
from nlp100pon_84 import CreateW2VEmbedding
from nlp100pon_86 import CNN
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

    epochs = 30
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    out_channels = int(trial.suggest_discrete_uniform('out_channels', 32, 1024, 32))
    emb_dim = int(trial.suggest_discrete_uniform('emb_dim', 32, 1024, 32))

    model = CNN(
        vocab_size=len(vocab), 
        emb_dim=emb_dim, 
        out_channels=out_channels, 
        out_dim=4,
        emb_weights=emb_weights,
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

    TRIAL_SIZE = 20
    study = optuna.create_study()
    study.optimize(objective, n_trials=TRIAL_SIZE)

    print('best parameters: {}'.format(study.best_params), end='\n\n')

    model = CNN(
        vocab_size=len(vocab), 
        emb_dim=int(study.best_params['emb_dim']), 
        out_channels=int(study.best_params['out_channels']), 
        out_dim=4,
        emb_weights=emb_weights,
    ).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.9)
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
[I 2022-06-22 11:44:34,498] A new study created in memory with name: no-name-8630aca8-7edd-44af-b4d5-79751ae66b78
[I 2022-06-22 11:44:49,954] Trial 0 finished with value: 0.986765282443617 and parameters: {'out_channels': 32.0, 'emb_dim': 256.0}. Best is trial 0 with value: 0.986765282443617.
[I 2022-06-22 11:45:59,061] Trial 1 finished with value: 1.3261209227692061 and parameters: {'out_channels': 576.0, 'emb_dim': 768.0}. Best is trial 0 with value: 0.986765282443617.
[I 2022-06-22 11:46:23,993] Trial 2 finished with value: 1.3261209227692061 and parameters: {'out_channels': 128.0, 'emb_dim': 736.0}. Best is trial 0 with value: 0.986765282443617.
[I 2022-06-22 11:46:44,628] Trial 3 finished with value: 1.3566614462696631 and parameters: {'out_channels': 640.0, 'emb_dim': 96.0}. Best is trial 0 with value: 0.986765282443617.
[I 2022-06-22 11:47:13,279] Trial 4 finished with value: 1.3566612003446519 and parameters: {'out_channels': 928.0, 'emb_dim': 128.0}. Best is trial 0 with value: 0.986765282443617.
[I 2022-06-22 11:48:50,328] Trial 5 finished with value: 1.3261209227692061 and parameters: {'out_channels': 704.0, 'emb_dim': 1024.0}. Best is trial 0 with value: 0.986765282443617.
[I 2022-06-22 11:49:32,805] Trial 6 finished with value: 1.3566612003446519 and parameters: {'out_channels': 224.0, 'emb_dim': 992.0}. Best is trial 0 with value: 0.986765282443617.
[I 2022-06-22 11:50:39,251] Trial 7 finished with value: 1.3261209227692061 and parameters: {'out_channels': 704.0, 'emb_dim': 608.0}. Best is trial 0 with value: 0.986765282443617.
[I 2022-06-22 11:50:55,917] Trial 8 finished with value: 1.3261209227692061 and parameters: {'out_channels': 480.0, 'emb_dim': 64.0}. Best is trial 0 with value: 0.986765282443617.
[I 2022-06-22 11:51:52,246] Trial 9 finished with value: 1.356661206063838 and parameters: {'out_channels': 416.0, 'emb_dim': 800.0}. Best is trial 0 with value: 0.986765282443617.
[I 2022-06-22 11:52:07,186] Trial 10 finished with value: 0.9881241439521938 and parameters: {'out_channels': 32.0, 'emb_dim': 352.0}. Best is trial 0 with value: 0.986765282443617.
[I 2022-06-22 11:52:22,056] Trial 11 finished with value: 1.0019783637691653 and parameters: {'out_channels': 32.0, 'emb_dim': 352.0}. Best is trial 0 with value: 0.986765282443617.
[I 2022-06-22 11:52:49,657] Trial 12 finished with value: 1.3261209170500199 and parameters: {'out_channels': 288.0, 'emb_dim': 352.0}. Best is trial 0 with value: 0.986765282443617.
[I 2022-06-22 11:53:04,426] Trial 13 finished with value: 0.9656422971070617 and parameters: {'out_channels': 32.0, 'emb_dim': 320.0}. Best is trial 13 with value: 0.9656422971070617.
[I 2022-06-22 11:53:23,884] Trial 14 finished with value: 0.9814352352937301 and parameters: {'out_channels': 256.0, 'emb_dim': 256.0}. Best is trial 13 with value: 0.9656422971070617.
[I 2022-06-22 11:53:52,780] Trial 15 finished with value: 1.3566612003446519 and parameters: {'out_channels': 320.0, 'emb_dim': 448.0}. Best is trial 13 with value: 0.9656422971070617.
[I 2022-06-22 11:54:10,247] Trial 16 finished with value: 0.9765662260498779 and parameters: {'out_channels': 192.0, 'emb_dim': 224.0}. Best is trial 13 with value: 0.9656422971070617.
[I 2022-06-22 11:54:34,923] Trial 17 finished with value: 0.9744264554048049 and parameters: {'out_channels': 160.0, 'emb_dim': 544.0}. Best is trial 13 with value: 0.9656422971070617.
[I 2022-06-22 11:55:11,085] Trial 18 finished with value: 1.3566612003446519 and parameters: {'out_channels': 384.0, 'emb_dim': 544.0}. Best is trial 13 with value: 0.9656422971070617.
[I 2022-06-22 11:55:33,818] Trial 19 finished with value: 1.3566592215061903 and parameters: {'out_channels': 128.0, 'emb_dim': 640.0}. Best is trial 13 with value: 0.9656422971070617.
best parameters: {'out_channels': 32.0, 'emb_dim': 320.0}

epoch: 1 Train [loss: 1.3255, accuracy: 0.4188], Valid [loss: 1.3261, accuracy: 0.4273]
epoch: 2 Train [loss: 1.3246, accuracy: 0.4210], Valid [loss: 1.3261, accuracy: 0.4273]
epoch: 3 Train [loss: 1.3248, accuracy: 0.4210], Valid [loss: 1.3261, accuracy: 0.4273]
epoch: 4 Train [loss: 1.3247, accuracy: 0.4210], Valid [loss: 1.3261, accuracy: 0.4273]
epoch: 5 Train [loss: 1.3247, accuracy: 0.4210], Valid [loss: 1.3261, accuracy: 0.4273]
epoch: 6 Train [loss: 1.3248, accuracy: 0.4210], Valid [loss: 1.3261, accuracy: 0.4273]
epoch: 7 Train [loss: 1.3246, accuracy: 0.4210], Valid [loss: 1.3261, accuracy: 0.4273]
epoch: 8 Train [loss: 1.3246, accuracy: 0.4210], Valid [loss: 1.3261, accuracy: 0.4273]
epoch: 9 Train [loss: 1.3245, accuracy: 0.4210], Valid [loss: 1.3261, accuracy: 0.4273]
epoch: 10 Train [loss: 1.3247, accuracy: 0.4210], Valid [loss: 1.3255, accuracy: 0.4273]
epoch: 11 Train [loss: 1.0623, accuracy: 0.6766], Valid [loss: 1.0286, accuracy: 0.7159]
epoch: 12 Train [loss: 0.9983, accuracy: 0.7445], Valid [loss: 1.0110, accuracy: 0.7354]
epoch: 13 Train [loss: 0.9791, accuracy: 0.7635], Valid [loss: 1.0007, accuracy: 0.7466]
epoch: 14 Train [loss: 0.9669, accuracy: 0.7766], Valid [loss: 1.0074, accuracy: 0.7421]
epoch: 15 Train [loss: 0.9606, accuracy: 0.7833], Valid [loss: 0.9897, accuracy: 0.7586]
epoch: 16 Train [loss: 0.9560, accuracy: 0.7879], Valid [loss: 0.9819, accuracy: 0.7676]
epoch: 17 Train [loss: 0.9511, accuracy: 0.7924], Valid [loss: 0.9951, accuracy: 0.7541]
epoch: 18 Train [loss: 0.9481, accuracy: 0.7956], Valid [loss: 0.9897, accuracy: 0.7609]
epoch: 19 Train [loss: 0.9465, accuracy: 0.7977], Valid [loss: 0.9873, accuracy: 0.7601]
epoch: 20 Train [loss: 0.9454, accuracy: 0.7989], Valid [loss: 0.9861, accuracy: 0.7639]
epoch: 21 Train [loss: 0.9439, accuracy: 0.7999], Valid [loss: 0.9884, accuracy: 0.7624]
epoch: 22 Train [loss: 0.9421, accuracy: 0.8020], Valid [loss: 0.9973, accuracy: 0.7519]
epoch: 23 Train [loss: 0.9419, accuracy: 0.8020], Valid [loss: 0.9838, accuracy: 0.7676]
epoch: 24 Train [loss: 0.9410, accuracy: 0.8033], Valid [loss: 1.0049, accuracy: 0.7451]
epoch: 25 Train [loss: 0.9399, accuracy: 0.8042], Valid [loss: 0.9885, accuracy: 0.7616]
epoch: 26 Train [loss: 0.9400, accuracy: 0.8045], Valid [loss: 0.9878, accuracy: 0.7616]
epoch: 27 Train [loss: 0.9382, accuracy: 0.8060], Valid [loss: 0.9940, accuracy: 0.7534]
epoch: 28 Train [loss: 0.9384, accuracy: 0.8057], Valid [loss: 0.9872, accuracy: 0.7624]
epoch: 29 Train [loss: 0.9360, accuracy: 0.8079], Valid [loss: 0.9798, accuracy: 0.7729]
epoch: 30 Train [loss: 0.9321, accuracy: 0.8109], Valid [loss: 0.9803, accuracy: 0.7691]

----------------
Train   [loss: 0.9165, accuracy: 0.8255]
Valid   [loss: 0.9731, accuracy: 0.7691]
Test    [loss: 0.9864, accuracy: 0.7526]
"""
