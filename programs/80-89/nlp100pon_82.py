from nlp100pon_80 import ConvertDatasetToIDSeqSet, LoadData
from nlp100pon_81 import RNN
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

def LoadTensorDatasetForRNN():

    from torch.nn.utils.rnn import pad_sequence
    from torch.utils.data import TensorDataset
    
    corpora_processed_dict, labels_dict, label_encoder = LoadData()
    id_sequence_set = ConvertDatasetToIDSeqSet(corpora_processed_dict, label_encoder)
    
    # return
    dataset = {}
    vocab = list(label_encoder.keys())

    for k in id_sequence_set.keys():
        # k ∈ {'train', 'valid', 'test'}
        features, labels = id_sequence_set[k], labels_dict[k]
        seq_lengths = torch.LongTensor(list(map(len, features)))
        seq_tensor = pad_sequence(features, batch_first=True)

        dataset[k] = TensorDataset(seq_tensor, seq_lengths, labels)
        
    return dataset, vocab, label_encoder
    



if __name__ == "__main__":
    
    dataset, vocab, _ = LoadTensorDatasetForRNN()

    """
    問83でミニバッチ化・GPU上での学習をするのであまり真面目にやらない
    データ数を300にして学習する
    """
    batch_size = 300
    train_dataloader    = DataLoader(dataset['train'], batch_size=batch_size, shuffle=False) #ミニバッチ学習時はshuffle=Trueにする
    valid_dataloader    = DataLoader(dataset['valid'], batch_size=batch_size, shuffle=False)
    test_dataloader     = DataLoader(dataset['test'], batch_size=batch_size, shuffle=False)
    train_size, valid_size, test_size = batch_size, batch_size, batch_size
    # print(len(dataset['train']), len(dataset['valid']), len(dataset['test']))
    
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
    epochs = 10

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
            
            break # データ数300でとめる
        
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

            break # データ数300でとめる
        
        valid_loss /= valid_size
        valid_acc = valid_num_correct / valid_size

        print('epoch: {} Train [loss: {:.4f}, accuracy: {:.4f}], Valid [loss: {:.4f}, accuracy: {:.4f}]'.
                format(epoch + 1, train_loss, train_acc, valid_loss, valid_acc))

"""
Using cuda device
epoch: 1 Train [loss: 1.3925, accuracy: 0.2600], Valid [loss: 1.3832, accuracy: 0.3300]
epoch: 2 Train [loss: 1.3840, accuracy: 0.2933], Valid [loss: 1.3758, accuracy: 0.3833]
epoch: 3 Train [loss: 1.3756, accuracy: 0.3400], Valid [loss: 1.3687, accuracy: 0.4167]
epoch: 4 Train [loss: 1.3674, accuracy: 0.3933], Valid [loss: 1.3619, accuracy: 0.4400]
epoch: 5 Train [loss: 1.3593, accuracy: 0.4100], Valid [loss: 1.3553, accuracy: 0.4433]
epoch: 6 Train [loss: 1.3513, accuracy: 0.4167], Valid [loss: 1.3490, accuracy: 0.4500]
epoch: 7 Train [loss: 1.3434, accuracy: 0.4533], Valid [loss: 1.3430, accuracy: 0.4333]
epoch: 8 Train [loss: 1.3356, accuracy: 0.4900], Valid [loss: 1.3372, accuracy: 0.4400]
epoch: 9 Train [loss: 1.3279, accuracy: 0.5400], Valid [loss: 1.3317, accuracy: 0.4367]
epoch: 10 Train [loss: 1.3203, accuracy: 0.5667], Valid [loss: 1.3264, accuracy: 0.4167]
"""