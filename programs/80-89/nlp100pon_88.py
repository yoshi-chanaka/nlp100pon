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

    hid_dim = int(trial.suggest_discrete_uniform('hid_dim', 16, 256, 16))
    num_layers = int(trial.suggest_discrete_uniform('num_layers', 1, 4, 1))

    model = BiRNN(
        vocab_size=len(vocab), 
        emb_dim=300, 
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
    emb_weights = CreateW2VEmbedding(w2vVocab) # Noneにするとword2vecの単語埋め込みは使わない

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

    model = BiRNN(
        vocab_size=len(vocab), 
        emb_dim=300, 
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
loading word2vec ...
creating embedding matrix ...
size of embedding weights: (8492, 300)
number of unregistered words: 928
data size: 10672, 1334, 1334
Using cuda device
[I 2022-06-10 17:30:38,966] A new study created in memory with name: no-name-2c213ba9-2ad8-4d1c-918f-6d134cc29f62
[I 2022-06-10 17:31:09,580] Trial 0 finished with value: 0.8480987891979304 and parameters: {'hid_dim': 16.0, 'num_layers': 2.0}. Best is trial 0 with value: 0.8480987891979304.
[I 2022-06-10 17:31:50,525] Trial 1 finished with value: 0.8999554406756582 and parameters: {'hid_dim': 112.0, 'num_layers': 3.0}. Best is trial 0 with value: 0.8480987891979304.
[I 2022-06-10 17:32:38,368] Trial 2 finished with value: 0.9132595291023312 and parameters: {'hid_dim': 144.0, 'num_layers': 4.0}. Best is trial 0 with value: 0.8480987891979304.
[I 2022-06-10 17:33:14,838] Trial 3 finished with value: 0.9036583821812848 and parameters: {'hid_dim': 16.0, 'num_layers': 3.0}. Best is trial 0 with value: 0.8480987891979304.
[I 2022-06-10 17:33:45,713] Trial 4 finished with value: 0.8502946936565897 and parameters: {'hid_dim': 240.0, 'num_layers': 2.0}. Best is trial 0 with value: 0.8480987891979304.
[I 2022-06-10 17:34:25,840] Trial 5 finished with value: 0.8976875659765332 and parameters: {'hid_dim': 224.0, 'num_layers': 3.0}. Best is trial 0 with value: 0.8480987891979304.
[I 2022-06-10 17:34:48,424] Trial 6 finished with value: 0.8334742979309906 and parameters: {'hid_dim': 48.0, 'num_layers': 1.0}. Best is trial 6 with value: 0.8334742979309906.
[I 2022-06-10 17:35:11,095] Trial 7 finished with value: 0.8359945403046157 and parameters: {'hid_dim': 112.0, 'num_layers': 1.0}. Best is trial 6 with value: 0.8334742979309906.
[I 2022-06-10 17:36:00,531] Trial 8 finished with value: 0.9131273787239681 and parameters: {'hid_dim': 192.0, 'num_layers': 4.0}. Best is trial 6 with value: 0.8334742979309906.
[I 2022-06-10 17:36:52,293] Trial 9 finished with value: 0.9253627742784492 and parameters: {'hid_dim': 96.0, 'num_layers': 4.0}. Best is trial 6 with value: 0.8334742979309906.
[I 2022-06-10 17:37:15,419] Trial 10 finished with value: 0.8382546597871109 and parameters: {'hid_dim': 64.0, 'num_layers': 1.0}. Best is trial 6 with value: 0.8334742979309906.
[I 2022-06-10 17:37:39,233] Trial 11 finished with value: 0.8399900210493508 and parameters: {'hid_dim': 64.0, 'num_layers': 1.0}. Best is trial 6 with value: 0.8334742979309906.
[I 2022-06-10 17:38:02,269] Trial 12 finished with value: 0.8419275298111443 and parameters: {'hid_dim': 144.0, 'num_layers': 1.0}. Best is trial 6 with value: 0.8334742979309906.
[I 2022-06-10 17:38:34,155] Trial 13 finished with value: 0.8637025567187719 and parameters: {'hid_dim': 64.0, 'num_layers': 2.0}. Best is trial 6 with value: 0.8334742979309906.
[I 2022-06-10 17:38:57,267] Trial 14 finished with value: 0.8396382310401196 and parameters: {'hid_dim': 176.0, 'num_layers': 1.0}. Best is trial 6 with value: 0.8334742979309906.
[I 2022-06-10 17:39:28,822] Trial 15 finished with value: 0.8731390225297507 and parameters: {'hid_dim': 96.0, 'num_layers': 2.0}. Best is trial 6 with value: 0.8334742979309906.
[I 2022-06-10 17:39:49,754] Trial 16 finished with value: 0.8393211850876929 and parameters: {'hid_dim': 48.0, 'num_layers': 1.0}. Best is trial 6 with value: 0.8334742979309906.
[I 2022-06-10 17:40:12,538] Trial 17 finished with value: 0.8382485831516615 and parameters: {'hid_dim': 96.0, 'num_layers': 1.0}. Best is trial 6 with value: 0.8334742979309906.
[I 2022-06-10 17:40:43,866] Trial 18 finished with value: 0.8461160044977512 and parameters: {'hid_dim': 128.0, 'num_layers': 2.0}. Best is trial 6 with value: 0.8334742979309906.
[I 2022-06-10 17:41:06,391] Trial 19 finished with value: 0.8363687981372473 and parameters: {'hid_dim': 176.0, 'num_layers': 1.0}. Best is trial 6 with value: 0.8334742979309906.
best parameters: {'hid_dim': 48.0, 'num_layers': 1.0}

epoch: 1 Train [loss: 1.0697, accuracy: 0.6957], Valid [loss: 0.9598, accuracy: 0.7909]
epoch: 2 Train [loss: 0.9633, accuracy: 0.7797], Valid [loss: 0.9524, accuracy: 0.7976]
epoch: 3 Train [loss: 0.9461, accuracy: 0.7937], Valid [loss: 0.9382, accuracy: 0.8043]
epoch: 4 Train [loss: 0.9215, accuracy: 0.8229], Valid [loss: 0.9259, accuracy: 0.8223]
epoch: 5 Train [loss: 0.8986, accuracy: 0.8434], Valid [loss: 0.8993, accuracy: 0.8456]
epoch: 6 Train [loss: 0.8783, accuracy: 0.8657], Valid [loss: 0.8806, accuracy: 0.8763]
epoch: 7 Train [loss: 0.8469, accuracy: 0.9018], Valid [loss: 0.8651, accuracy: 0.8853]
epoch: 8 Train [loss: 0.8295, accuracy: 0.9160], Valid [loss: 0.8506, accuracy: 0.9025]
epoch: 9 Train [loss: 0.8157, accuracy: 0.9316], Valid [loss: 0.8625, accuracy: 0.8853]
epoch: 10 Train [loss: 0.8089, accuracy: 0.9382], Valid [loss: 0.8398, accuracy: 0.9123]
epoch: 11 Train [loss: 0.8020, accuracy: 0.9442], Valid [loss: 0.8380, accuracy: 0.9115]
epoch: 12 Train [loss: 0.7989, accuracy: 0.9471], Valid [loss: 0.8368, accuracy: 0.9145]
epoch: 13 Train [loss: 0.7951, accuracy: 0.9515], Valid [loss: 0.8354, accuracy: 0.9160]
epoch: 14 Train [loss: 0.7941, accuracy: 0.9527], Valid [loss: 0.8412, accuracy: 0.9070]
epoch: 15 Train [loss: 0.7873, accuracy: 0.9591], Valid [loss: 0.8644, accuracy: 0.8846]
epoch: 16 Train [loss: 0.7856, accuracy: 0.9605], Valid [loss: 0.8352, accuracy: 0.9130]
epoch: 17 Train [loss: 0.7836, accuracy: 0.9619], Valid [loss: 0.8414, accuracy: 0.9078]
epoch: 18 Train [loss: 0.7814, accuracy: 0.9646], Valid [loss: 0.8356, accuracy: 0.9123]
epoch: 19 Train [loss: 0.7798, accuracy: 0.9660], Valid [loss: 0.8336, accuracy: 0.9138]
epoch: 20 Train [loss: 0.7776, accuracy: 0.9682], Valid [loss: 0.8333, accuracy: 0.9160]
epoch: 21 Train [loss: 0.7773, accuracy: 0.9687], Valid [loss: 0.8352, accuracy: 0.9153]
epoch: 22 Train [loss: 0.7766, accuracy: 0.9689], Valid [loss: 0.8421, accuracy: 0.9055]
epoch: 23 Train [loss: 0.7755, accuracy: 0.9700], Valid [loss: 0.8402, accuracy: 0.9063]
epoch: 24 Train [loss: 0.7743, accuracy: 0.9710], Valid [loss: 0.8359, accuracy: 0.9145]
epoch: 25 Train [loss: 0.7736, accuracy: 0.9714], Valid [loss: 0.8357, accuracy: 0.9130]
epoch: 26 Train [loss: 0.7734, accuracy: 0.9716], Valid [loss: 0.8332, accuracy: 0.9160]
epoch: 27 Train [loss: 0.7728, accuracy: 0.9722], Valid [loss: 0.8357, accuracy: 0.9145]
epoch: 28 Train [loss: 0.7724, accuracy: 0.9726], Valid [loss: 0.8388, accuracy: 0.9100]
epoch: 29 Train [loss: 0.7724, accuracy: 0.9726], Valid [loss: 0.8339, accuracy: 0.9145]
epoch: 30 Train [loss: 0.7721, accuracy: 0.9728], Valid [loss: 0.8330, accuracy: 0.9160]

----------------
Train   [loss: 0.7703, accuracy: 0.9735]
Valid   [loss: 0.8264, accuracy: 0.9160]
Test    [loss: 0.8307, accuracy: 0.9123]
"""
