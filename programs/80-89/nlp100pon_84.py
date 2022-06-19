from nlp100pon_82 import LoadTensorDatasetForRNN
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence

def CreateW2VEmbedding(
    w2vVocab, 
    w2v_path = '../../data/GoogleNews-vectors-negative300.bin',
    dim = 300, 
):
    
    from gensim.models import KeyedVectors
    import numpy as np

    print('loading word2vec ... ')
    w2v_model = KeyedVectors.load_word2vec_format(w2v_path, binary=True)

    np.random.seed(0)
    emb_weights = np.random.rand(1, dim) # ラベル0
    cnt_unregistered = 0
    print('creating embedding matrix ... ')
    for i, w in enumerate(w2vVocab):
        try:
            emb_weights = np.vstack([emb_weights, w2v_model[w]])
        except:
            np.random.seed(i)
            emb_weights = np.vstack([emb_weights, np.random.rand(1, dim)])
            cnt_unregistered += 1

    print('size of embedding weights: {}'.format(emb_weights.shape))
    print('number of unregistered words: {}'.format(cnt_unregistered))
    return torch.from_numpy(emb_weights.astype(np.float32))
    

class RNN_w2vEmbed(nn.Module):

    def __init__(self, vocab_size, emb_dim, hid_dim, out_dim, emb_weights):
        super(RNN_w2vEmbed, self).__init__()
        self.emb = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_dim)
        self.emb.weight = nn.Parameter(emb_weights)
        self.rnn = nn.LSTM(
            input_size=emb_dim, 
            hidden_size=hid_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False
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
    emb_weights = CreateW2VEmbedding(w2vVocab)

    batch_size = 64
    train_dataloader    = DataLoader(dataset['train'], batch_size=batch_size, shuffle=True)
    valid_dataloader    = DataLoader(dataset['valid'], batch_size=batch_size, shuffle=False)
    test_dataloader     = DataLoader(dataset['test'], batch_size=batch_size, shuffle=False)
    train_size, valid_size, test_size = len(dataset['train']), len(dataset['valid']), len(dataset['test'])
    print('data size: {}, {}, {}'.format(train_size, valid_size, test_size))
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} device'.format(device))

    model = RNN_w2vEmbed(
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
epoch: 1 Train [loss: 1.0999, accuracy: 0.6516], Valid [loss: 0.9610, accuracy: 0.7909]
epoch: 2 Train [loss: 0.9632, accuracy: 0.7803], Valid [loss: 0.9678, accuracy: 0.7834]
epoch: 3 Train [loss: 0.9483, accuracy: 0.7919], Valid [loss: 0.9393, accuracy: 0.8043]
epoch: 4 Train [loss: 0.9250, accuracy: 0.8179], Valid [loss: 0.9278, accuracy: 0.8171]
epoch: 5 Train [loss: 0.8996, accuracy: 0.8453], Valid [loss: 0.9101, accuracy: 0.8448]
epoch: 6 Train [loss: 0.8692, accuracy: 0.8841], Valid [loss: 0.8826, accuracy: 0.8636]
epoch: 7 Train [loss: 0.8469, accuracy: 0.8984], Valid [loss: 0.8630, accuracy: 0.8898]
epoch: 8 Train [loss: 0.8273, accuracy: 0.9187], Valid [loss: 0.8544, accuracy: 0.8951]
epoch: 9 Train [loss: 0.8172, accuracy: 0.9291], Valid [loss: 0.8478, accuracy: 0.9018]
epoch: 10 Train [loss: 0.8050, accuracy: 0.9410], Valid [loss: 0.8427, accuracy: 0.9078]
epoch: 11 Train [loss: 0.8003, accuracy: 0.9462], Valid [loss: 0.8409, accuracy: 0.9063]
epoch: 12 Train [loss: 0.7942, accuracy: 0.9520], Valid [loss: 0.8418, accuracy: 0.9100]
epoch: 13 Train [loss: 0.7963, accuracy: 0.9497], Valid [loss: 0.8442, accuracy: 0.9100]
epoch: 14 Train [loss: 0.7888, accuracy: 0.9566], Valid [loss: 0.8371, accuracy: 0.9138]
epoch: 15 Train [loss: 0.7849, accuracy: 0.9611], Valid [loss: 0.8372, accuracy: 0.9108]
epoch: 16 Train [loss: 0.7818, accuracy: 0.9639], Valid [loss: 0.8746, accuracy: 0.8748]
epoch: 17 Train [loss: 0.7809, accuracy: 0.9650], Valid [loss: 0.8400, accuracy: 0.9078]
epoch: 18 Train [loss: 0.7788, accuracy: 0.9671], Valid [loss: 0.8377, accuracy: 0.9123]
epoch: 19 Train [loss: 0.7773, accuracy: 0.9680], Valid [loss: 0.8363, accuracy: 0.9123]
epoch: 20 Train [loss: 0.7766, accuracy: 0.9686], Valid [loss: 0.8393, accuracy: 0.9085]
epoch: 21 Train [loss: 0.7758, accuracy: 0.9695], Valid [loss: 0.8376, accuracy: 0.9093]
epoch: 22 Train [loss: 0.7748, accuracy: 0.9705], Valid [loss: 0.8377, accuracy: 0.9093]
epoch: 23 Train [loss: 0.7753, accuracy: 0.9698], Valid [loss: 0.8355, accuracy: 0.9145]
epoch: 24 Train [loss: 0.7739, accuracy: 0.9711], Valid [loss: 0.8369, accuracy: 0.9115]
epoch: 25 Train [loss: 0.7735, accuracy: 0.9715], Valid [loss: 0.8365, accuracy: 0.9123]
epoch: 26 Train [loss: 0.7734, accuracy: 0.9716], Valid [loss: 0.8366, accuracy: 0.9130]
epoch: 27 Train [loss: 0.7735, accuracy: 0.9713], Valid [loss: 0.8363, accuracy: 0.9115]
epoch: 28 Train [loss: 0.7730, accuracy: 0.9719], Valid [loss: 0.8365, accuracy: 0.9108]
epoch: 29 Train [loss: 0.7729, accuracy: 0.9720], Valid [loss: 0.8372, accuracy: 0.9123]
epoch: 30 Train [loss: 0.7728, accuracy: 0.9720], Valid [loss: 0.8373, accuracy: 0.9123]

----------------
Train   [loss: 0.7716, accuracy: 0.9721]
Valid   [loss: 0.8307, accuracy: 0.9123]
Test    [loss: 0.8316, accuracy: 0.9123]
"""
