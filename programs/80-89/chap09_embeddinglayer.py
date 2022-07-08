from nlp100pon_80 import ConvertDatasetToIDSeqSet, LoadData

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence


class Embed(nn.Module):

    def __init__(self, vocab_size, emb_dim):
        super(Embed, self).__init__()
        self.emb = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_dim)

    def forward(self, x):
        return self.emb(x)

if __name__ == "__main__":
    
    corpora_processed_dict, labels_dict, label_encoder = LoadData()
    vocab = list(label_encoder.keys())
    id_sequence_set = ConvertDatasetToIDSeqSet(corpora_processed_dict, label_encoder)

    data = id_sequence_set['train'][:3]
    labels_true = labels_dict['train'][:3]
    seq_lengths = torch.LongTensor(list(map(len, data)))
    seq_tensor = pad_sequence(data, batch_first=True) # パディング

    device = 'cpu' # 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} device'.format(device))

    model = Embed(
        vocab_size=len(vocab), 
        emb_dim=300, 
    ).to(device)

    seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
    seq_tensor = seq_tensor[perm_idx].to(device)
    labels_true = labels_true[perm_idx]

    output = model(seq_tensor).cpu()
    print(output.shape)
    print(data[0])
    print(output[0])

"""
Using cpu device
torch.Size([3, 11, 300])
tensor([ 142,  207, 3231,    0,  556, 2349, 1262,    0,  623,    0, 1534])
tensor([[ 0.2719,  0.1176, -1.2007,  ..., -0.4455,  0.2564,  0.5492],
        [ 0.4109,  0.3876, -0.1650,  ...,  0.4189, -0.2210,  0.6142],
        [-0.8373,  0.6231,  1.6986,  ...,  1.7157,  1.2358, -1.3393],
        ...,
        [ 0.7176, -1.3824,  1.4526,  ..., -1.3017,  1.6922, -1.0173],
        [ 0.6792,  0.3425, -2.3978,  ...,  0.1640, -0.2659,  1.3363],
        [ 1.1705,  0.0505, -2.2676,  ...,  2.0386,  0.6396, -2.5864]],
       grad_fn=<SelectBackward0>)
"""

