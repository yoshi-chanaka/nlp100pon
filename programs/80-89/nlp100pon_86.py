from nlp100pon_82 import LoadTensorDatasetForRNN
from nlp100pon_84 import CreateW2VEmbedding
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

class CNN(nn.Module):

    def __init__(self, vocab_size, emb_dim, out_channels, out_dim, emb_weights=None):
        super(CNN, self).__init__()
        self.emb = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_dim)
        if emb_weights != None:
            self.emb.weight = nn.Parameter(emb_weights)
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=out_channels, 
            kernel_size=(3, emb_dim), # [emb(x_(t-1));emb(x_t);emb(x_(t+1)))]
            stride=1, 
            padding=(1, 0)
        )
        self.linear = nn.Linear(out_channels, out_dim)

    def print_forward(self, x):
        print(x.shape)
        pool_kernel_size = x.shape[1]   # (batch_size, max_length)
        x = self.emb(x).unsqueeze(1)    # (batch_size, 1, max_length, emb_dim)
        print('emb:\t', x.shape)
        x = self.conv(x).relu()         # (batch_size, out_channels, max_length, 1)
        print('conv:\t', x.shape)
        x = F.max_pool2d(x, kernel_size=(pool_kernel_size, 1), stride=1) # (batch_size, out_channels, 1, 1)
        print('pool:\t', x.shape)
        x = torch.flatten(x, 1)         # (batch_size, out_channels)
        print('flatten:\t', x.shape)
        x = self.linear(x)              # (batch_size, out_dim)
        print('linear:', x.shape)

        return F.softmax(x, dim=1)
    
    def forward(self, x, seq_lengths=None):
        pool_kernel_size = x.shape[1]   # (batch_size, max_length)
        x = self.emb(x).unsqueeze(1)    # (batch_size, 1, max_length, emb_dim)
        x = self.conv(x).relu()         # (batch_size, out_channels, max_length, 1)
        x = F.max_pool2d(
            x, kernel_size=(pool_kernel_size, 1), stride=1
        )                               # (batch_size, out_channels, 1, 1)
        x = torch.flatten(x, 1)         # (batch_size, out_channels)
        x = self.linear(x)              # (batch_size, out_dim)

        return F.softmax(x, dim=1)

if __name__ == "__main__":

    dataset, vocab, label_encoder = LoadTensorDatasetForRNN()
    # 単語埋め込み行列の作成
    w2vVocab = [word for word, idx in label_encoder.items() if idx > 0]
    emb_weights = None # CreateW2VEmbedding(w2vVocab)
    
    batch_size = 3
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

    for X, seq_lengths, y in train_dataloader:

        output = model.print_forward(X.to(device)).cpu()
        print(output, end='\n\n')
        print('pred: ', output.argmax(1))
        print('true: ', y) 
        
        break

"""
data size: 10672, 1334, 1334
Using cuda device
torch.Size([3, 75])
emb:     torch.Size([3, 1, 75, 300])
conv:    torch.Size([3, 128, 75, 1])
pool:    torch.Size([3, 128, 1, 1])
flatten:         torch.Size([3, 128])
linear: torch.Size([3, 4])
tensor([[0.3760, 0.2663, 0.2722, 0.0855],
        [0.4243, 0.1963, 0.2949, 0.0845],
        [0.4163, 0.2436, 0.2598, 0.0802]], grad_fn=<ToCopyBackward0>)

pred:  tensor([0, 0, 0])
true:  tensor([0, 2, 2])
"""
