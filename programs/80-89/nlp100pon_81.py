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

"""
Using cuda device
tensor([[0.2318, 0.2748, 0.2114, 0.2820],
        [0.2313, 0.2401, 0.2397, 0.2888],
        [0.2347, 0.3009, 0.1727, 0.2917],
        [0.2047, 0.2574, 0.2020, 0.3359],
        [0.2399, 0.2802, 0.2270, 0.2528]], grad_fn=<ToCopyBackward0>)

pred:  tensor([3, 3, 1, 3, 1])
true:  tensor([2, 0, 0, 2, 0])
"""

