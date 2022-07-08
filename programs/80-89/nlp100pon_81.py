from nlp100pon_80 import ConvertDatasetToIDSeqSet, LoadData

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence


class RNN(nn.Module):

    def __init__(self, vocab_size, emb_dim, hid_dim, out_dim):
        super(RNN, self).__init__()
        self.emb = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_dim)
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
        # print(ht[-1])
        # output, input_sizes = pad_packed_sequence(packed_output, batch_first=True)
        # print(output)
        h = self.linear(ht).squeeze(0)
        return F.softmax(h, dim=1)
    
    def forward_visible(self, x, seq_lengths):
        x = self.emb(x)
        x = pack_padded_sequence(x, seq_lengths.cpu().numpy(), batch_first=True)
        packed_output, (ht, ct) = self.rnn(x)
        print(packed_output, end='\n========\n')
        print(ht, end='\n========\n')
        print(ct, end='\n========\n')
        h = self.linear(ht).squeeze(0)
        print(h, end='\n========\n')
        return F.softmax(h, dim=1)

if __name__ == "__main__":
    """
    参考: 
    https://cod-aid.com/pytorch-pack
    https://gist.github.com/HarshTrivedi/f4e7293e941b17d19058f6fb90ab0fec
    https://hilinker.hatenablog.com/entry/2018/06/23/204910
    """
    corpora_processed_dict, labels_dict, label_encoder = LoadData()
    vocab = list(label_encoder.keys())
    id_sequence_set = ConvertDatasetToIDSeqSet(corpora_processed_dict, label_encoder)

    data = id_sequence_set['train'][:5]
    labels_true = labels_dict['train'][:5]
    seq_lengths = torch.LongTensor(list(map(len, data)))
    seq_tensor = pad_sequence(data, batch_first=True) # パディング

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} device'.format(device))

    model = RNN(
        vocab_size=len(vocab), 
        emb_dim=300, 
        hid_dim=8, 
        out_dim=4
    ).to(device)

    seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
    seq_tensor = seq_tensor[perm_idx].to(device)
    labels_true = labels_true[perm_idx]

    output = model(seq_tensor, seq_lengths).cpu()
    print(output, end='\n\n')
    print('pred: ', output.argmax(1))
    print('true: ', labels_true)    

    # output = model.forward_visible(seq_tensor, seq_lengths).cpu()

"""
Using cuda device
tensor([[0.2104, 0.3492, 0.1989, 0.2416],
        [0.2719, 0.2796, 0.2375, 0.2110],
        [0.2454, 0.2214, 0.2824, 0.2508],
        [0.1683, 0.2432, 0.2696, 0.3189],
        [0.2100, 0.2709, 0.2724, 0.2467]], grad_fn=<ToCopyBackward0>)

pred:  tensor([1, 1, 2, 3, 2])
true:  tensor([2, 0, 0, 2, 0])
"""

