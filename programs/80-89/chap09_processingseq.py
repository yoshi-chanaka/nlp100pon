from nlp100pon_80 import ConvertDatasetToIDSeqSet, LoadData

import torch
from torch.nn import Embedding, LSTM
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence


if __name__ == "__main__":
    """
    参考: 
    https://gist.github.com/HarshTrivedi/f4e7293e941b17d19058f6fb90ab0fec
    """
    
    corpora_processed_dict, labels_dict, label_encoder = LoadData()
    vocab = list(label_encoder.keys())
    id_sequence_set = ConvertDatasetToIDSeqSet(corpora_processed_dict, label_encoder)

    data = id_sequence_set['train'][:5]
    print('original\n', data, end='\n\n')
    seq_lengths = torch.LongTensor(list(map(len, data)))
    seq_tensor = pad_sequence(data, batch_first=True)
    print('pad_sequence\n', seq_tensor, end='\n\n')
    print('seq_lengths\n', seq_lengths, end='\n\n')
    
    seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
    seq_tensor = seq_tensor[perm_idx]
    
    embed = Embedding(len(vocab), 4) # embedding_dim = 4
    lstm = LSTM(input_size=4, hidden_size=5, batch_first=True)

    embedded_seq_tensor = embed(seq_tensor)
    print('embedded_seq_tensor\n', embedded_seq_tensor, end='\n\n')

    packed_input = pack_padded_sequence(embedded_seq_tensor, seq_lengths.cpu().numpy(), batch_first=True)
    print('packed_input\n', packed_input, end='\n\n')

    packed_output, (ht, ct) = lstm(packed_input)
    output, input_sizes = pad_packed_sequence(packed_output, batch_first=True)
    print('output\n', output, end='\n\n')

    print(ht[-1])







