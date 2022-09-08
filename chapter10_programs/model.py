from mask import create_mask
from load import load_data

import torch
from torch import Tensor
import torch.nn as nn
import math
from torch.utils.data import DataLoader

"""
参考: LANGUAGE TRANSLATION WITH NN.TRANSFORMER AND TORCHTEXT
https://pytorch.org/tutorials/beginner/translation_transformer.html
"""


class PositionalEncoding(nn.Module):
    def __init__(
        self,
        emb_size: int,
        dropout: float = 0.1,
        maxlen: int = 5000
    ):

        super(PositionalEncoding, self).__init__()
        den = torch.exp(
            - torch.arange(0, emb_size, 2) * math.log(10000) / emb_size
        )
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(
            token_embedding + self.pos_embedding[:token_embedding.size(0), :]
        )


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        # math.sqrt(self.emb_size) の役割？
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


class TransformerNMTModel(nn.Module):

    def __init__(
        self,
        num_encoder_layers: int,
        num_decoder_layers: int,
        emb_size: int,
        src_vocab_size: int,
        tgt_vocab_size: int,
        nhead: int = 8,
        dim_feedforward: int = 512,
        dropout: float = 0.1
    ):

        super(TransformerNMTModel, self).__init__()
        """
        torch.nn.Transformer
        https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html
        """
        self.transformer = nn.Transformer(
            d_model=emb_size,  # the number of expected features in the encoder/decoder inputs
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.generator = nn.Linear(emb_size, tgt_vocab_size)  # トークンの予測分布を出す
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(
            emb_size, dropout=dropout)

    def forward(
        self,
        src: Tensor,
        trg: Tensor,
        src_mask: Tensor,
        tgt_mask: Tensor,
        src_padding_mask: Tensor,
        tgt_padding_mask: Tensor,
        memory_key_padding_mask: Tensor
    ):

        src = src.transpose(0, 1)
        trg = trg.transpose(0, 1)
        src_mask, tgt_mask = src_mask[0], tgt_mask[0]
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        outs = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            memory_mask=None,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )
        return self.generator(outs).transpose(0, 1)

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer.encoder(
            self.positional_encoding(self.src_tok_emb(src)), src_mask
        )

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer.decoder(
            self.positional_encoding(self.tgt_tok_emb(tgt)), memory, tgt_mask
        )

    def forward_print(
        self,
        src: Tensor,
        trg: Tensor,
        src_mask: Tensor,
        tgt_mask: Tensor,
        src_padding_mask: Tensor,
        tgt_padding_mask: Tensor,
        memory_key_padding_mask: Tensor
    ):

        src = src.transpose(0, 1)
        trg = trg.transpose(0, 1)
        src_mask, tgt_mask = src_mask[0], tgt_mask[0]
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        print(f'positional enc: {src_emb.shape}, {tgt_emb.shape}')
        print(src_emb, tgt_emb)
        outs = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            memory_mask=None,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )
        print(f'\ntransformer output: {outs.shape}')
        return self.generator(outs).transpose(0, 1)


if __name__ == "__main__":

    dataset, vocab_src, vocab_tgt = load_data()
    batch_size = 2
    train_dataloader = DataLoader(dataset['train'], batch_size=batch_size, shuffle=True)
    # valid_dataloader    = DataLoader(dataset['dev'], batch_size=batch_size, shuffle=False)
    # test_dataloader     = DataLoader(dataset['test'], batch_size=batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using {} device'.format(device))

    SRC_VOCAB_SIZE = len(vocab_src)
    TGT_VOCAB_SIZE = len(vocab_tgt)
    EMB_SIZE = 512
    NHEAD = 8
    FFN_HID_DIM = 512
    BATCH_SIZE = 128
    NUM_ENCODER_LAYERS = 3
    NUM_DECODER_LAYERS = 3

    pad_idx = 1

    model = TransformerNMTModel(
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        emb_size=EMB_SIZE,
        src_vocab_size=SRC_VOCAB_SIZE,
        tgt_vocab_size=TGT_VOCAB_SIZE,
        nhead=NHEAD,
        dim_feedforward=FFN_HID_DIM,
    )
    """for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)"""
    model = model.to(device)
    path = '../models/model.pth'
    # model.load_state_dict(torch.load(path, map_location=device))
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_idx)

    for src, tgt in train_dataloader:

        src = src.transpose(0, 1).to(device)
        tgt = tgt.transpose(0, 1).to(device)

        tgt_input, tgt_output = tgt[:-1, :], tgt[1:, :]
        print('src/tgt shape:\t', src.shape, tgt.shape)

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)
        src_mask, tgt_mask = src_mask.to(device), tgt_mask.to(device)
        print('mask shape:\t', src_mask.shape, tgt_mask.shape)
        print('padding mask shape:\t', src_padding_mask.shape, tgt_padding_mask.shape)
        
        src_tr, tgt_input_tr = src.transpose(0, 1), tgt_input.transpose(0, 1)
        with torch.no_grad():
            pred = model.forward_print(
                    src=src_tr,
                    trg=tgt_input_tr,
                    src_mask=src_mask.repeat(src_tr.shape[0], 1, 1),
                    tgt_mask=tgt_mask.repeat(src_tr.shape[0], 1, 1),
                    src_padding_mask=src_padding_mask,
                    tgt_padding_mask=tgt_padding_mask,
                    memory_key_padding_mask=src_padding_mask
            )
        pred = pred.transpose(0, 1).reshape(-1, pred.shape[-1])
        print('predicted:\t', pred)
        print('predicted shape\t', pred.shape)
        print(torch.sum(pred.reshape(-1, pred.shape[-1]), axis=1))

        loss = loss_fn(pred.reshape(-1, pred.shape[-1])[:30], tgt_output.reshape(-1)[:30].long())
        print('loss:\t', loss)
        break

"""
Loading Dataset ...
Done
Using cuda device
src/tgt shape:   torch.Size([152, 2]) torch.Size([152, 2])
mask shape:      torch.Size([152, 152]) torch.Size([151, 151])
padding mask shape:      torch.Size([2, 152]) torch.Size([2, 151])
positional enc: torch.Size([152, 2, 512]), torch.Size([151, 2, 512])
tensor([[[-0.6193,  0.0000,  0.6035,  ...,  1.0441,  0.5135,  1.2948],
         [-0.6193,  1.0314,  0.6035,  ...,  1.0441,  0.5135,  1.2948]],

        [[ 0.7326,  0.0000,  0.0000,  ...,  0.6823,  0.6662,  1.4398],
         [ 1.1292,  0.8424,  0.2641,  ...,  0.9755, -0.2871,  1.4221]],

        [[ 1.1336, -0.7913,  0.0000,  ...,  0.7686,  0.5625,  0.9147],
         [ 0.3922, -0.2756,  1.6745,  ...,  0.8795,  0.4352,  1.6104]],

        ...,

        [[-1.6782,  0.1633,  0.0000,  ...,  0.0000,  0.4512,  1.5973],
         [ 0.0000,  0.1633, -0.6749,  ...,  1.2923,  0.4512,  0.0000]],

        [[-1.3896,  1.1889,  0.3109,  ...,  1.2923,  0.4514,  1.5973],
         [-1.3896,  1.1889,  0.3109,  ...,  1.2923,  0.4514,  0.0000]],

        [[-0.3707,  1.5001,  1.1199,  ...,  1.2923,  0.0000,  1.5973],
         [-0.3707,  1.5001,  1.1199,  ...,  0.0000,  0.0000,  1.5973]]],
       device='cuda:0') tensor([[[-0.4607,  1.3664, -0.5677,  ...,  0.5731, -0.6245,  1.6811],
         [-0.4607,  1.3664, -0.5677,  ...,  0.5731, -0.6245,  1.6811]],

        [[ 0.6923,  0.6791,  0.6485,  ...,  0.7892, -0.0947,  1.3144],
         [ 0.4206,  0.5664,  1.4162,  ...,  0.0000,  0.2321,  0.7743]],

        [[ 0.8966, -0.6175,  0.0000,  ...,  0.7959,  0.4271,  0.0000],
         [ 1.3183,  0.0403,  1.4189,  ...,  0.8663,  0.6100,  0.9415]],

        ...,

        [[ 0.0000, -1.6937, -0.8088,  ...,  1.5705,  0.2971,  0.0000],
         [-0.4253, -1.6937, -0.8088,  ...,  1.5705,  0.2971,  0.4907]],

        [[-1.1323, -0.8967,  0.0000,  ...,  1.5705,  0.2972,  0.4907],
         [-1.1323, -0.8967, -0.4945,  ...,  1.5705,  0.2972,  0.0000]],

        [[-0.8437,  0.1288,  0.4913,  ...,  1.5705,  0.2973,  0.4907],
         [-0.8437,  0.1288,  0.4913,  ...,  1.5705,  0.2973,  0.4907]]],
       device='cuda:0')

transformer output: torch.Size([151, 2, 512])
predicted:       tensor([[ 0.1845,  0.6649, -0.0290,  ..., -0.0676, -0.4308, -0.5124],
        [ 0.6610,  0.2846,  0.0953,  ...,  0.1582,  0.3074, -0.0466],
        [ 0.2244,  0.7012, -0.1164,  ...,  0.3005,  0.1259, -0.2380],
        ...,
        [ 0.4162,  0.5804, -0.2445,  ...,  0.3782, -0.1217, -0.2257],
        [ 0.3701,  0.6758, -0.0907,  ...,  0.1122, -0.1404, -0.0681],
        [ 0.3912,  0.7167, -0.3364,  ...,  0.2677, -0.0419, -0.0193]],
       device='cuda:0')
predicted shape  torch.Size([302, 8000])
tensor([-5.6193e+01, -3.7150e+01, -2.4318e+01, -2.3669e+01, -5.3240e+01,
        -4.7350e+01, -5.3436e+01, -6.2927e+01, -3.1220e+01, -4.1698e+01,
        -5.4336e+01, -3.0267e+01, -4.7245e+01, -4.0826e+01, -2.9672e+01,
        -1.0593e+01, -1.5944e+01, -1.7839e+01, -1.6446e+01, -8.3667e+00,
        -6.1001e+00, -4.1829e+01, -2.6416e+01, -4.4479e+01, -3.9340e+01,
        -2.5526e+01, -2.4208e+01, -4.8589e+01, -3.3416e+01, -3.8679e+01,
        -5.6778e+01, -7.0408e+00, -3.7992e+01, -3.8453e+01, -1.7580e+01,
        -3.0567e+01, -3.9649e+01, -3.4467e+01, -3.2762e+01, -2.7760e+01,
        -1.0273e+01, -3.6734e+01, -3.0260e+01, -6.0444e+01, -4.0254e+01,
        -4.9879e+01, -1.8304e+01, -1.8071e+01, -4.1917e+01, -2.6479e+01,
        -2.0508e+01, -4.3187e+00, -3.8748e+01, -1.1599e+01, -3.1964e+00,
        -3.1771e+01, -1.1864e+01, -2.2498e+01, -3.6270e+01, -1.7723e+01,
        -3.8093e+01,  1.2034e+01,  6.9275e-03, -3.1380e+01, -2.5007e+01,
        -4.3785e+00, -4.5512e+01, -3.2424e+01,  4.2180e+00, -2.2479e+01,
        -4.6140e+01, -4.3414e+01, -2.1585e+01, -2.7304e+01, -2.1813e+01,
        -7.0189e+01, -5.7698e+01, -5.8421e+01, -6.4252e+01, -4.7169e+01,
        -3.3496e+01, -5.4397e+01, -4.6277e+01, -4.5265e+01, -2.3560e+01,
        -6.2983e+01, -3.2382e+01, -1.1153e+01, -1.9523e+01, -2.3825e+01,
        -3.4747e+01, -1.8192e+01, -2.7231e+01, -3.4925e+01, -4.2492e+01,
        -2.0410e+01, -5.0686e+01, -4.6246e+01, -6.2844e+01, -5.2692e+01,
        -3.0523e+01, -4.0739e+01, -5.4763e+01, -5.6311e+01,  6.1520e+00,
         6.3253e+00, -2.4984e+01, -2.3257e+01, -5.2857e+01, -3.1767e+01,
        -2.8347e+01, -2.7756e+01, -2.6569e+01, -1.9210e+01, -6.0191e+01,
        -6.0901e+01, -2.3833e+01, -3.7920e+01, -2.8505e+01, -5.2308e+01,
        -6.2740e+01, -3.6275e+01, -5.5541e+01, -3.6666e+01, -3.6152e+01,
        -5.3225e+01, -3.9560e+01, -1.3807e+01, -3.1971e+01, -5.8669e+01,
        -1.7268e+01, -3.9549e+01, -2.6191e+01, -1.8057e+01, -3.8454e+01,
        -3.6833e+01, -1.9732e+01, -3.1629e+01, -5.3738e+01, -2.8792e+01,
        -3.3943e+01, -2.0024e+01, -5.7982e+01, -4.9817e+01, -3.8436e+01,
        -4.3512e+01, -5.0348e+01, -7.6036e+01, -5.4756e+01, -3.8611e+01,
        -5.6855e+01, -5.9506e+01, -4.9264e+01, -4.7494e+01, -6.9832e+01,
        -5.8619e+01, -3.5104e+01, -5.3968e+01, -5.8606e+01, -3.8359e+01,
        -5.5761e+01, -5.5259e+01, -5.4070e+01, -5.7958e+01, -5.1908e+01,
        -7.4567e+01, -3.6364e+01, -3.0677e+01, -5.9198e+01, -3.7037e+01,
        -6.0971e+01, -3.2990e+01, -3.6685e+01, -6.3296e+01, -4.7100e+01,
        -3.5960e+01, -5.3639e+01, -4.5689e+01, -3.3115e+01, -5.3101e+01,
        -5.8133e+01, -4.3662e+01, -3.7140e+01, -1.6341e+01, -5.3997e+01,
        -5.2229e+01, -2.6919e+01, -4.0223e+01, -2.9274e+01, -5.6128e+01,
        -2.2147e+01, -5.0845e+01, -3.5748e+01, -6.4274e+01, -5.0858e+01,
        -5.6257e+01, -2.3479e+01, -1.5355e+01, -5.2259e+01, -2.5887e+01,
        -2.8116e+01, -4.2990e+01, -5.9151e+01, -2.6901e+01, -5.4873e+01,
        -2.1815e+01, -3.0554e+01, -6.5160e+01, -4.2522e+01, -5.2411e+01,
        -4.3013e+01, -5.7135e+01, -2.9983e+01, -5.5151e+01, -4.0644e+01,
        -3.0931e+01, -3.0741e+01, -1.7295e+01, -4.1164e+01, -4.2774e+01,
        -4.8109e+01, -5.2411e+01, -2.0802e+01, -4.4683e+01, -5.1924e+01,
        -3.0982e+01, -5.4310e+01, -5.2227e+01, -6.7660e+01, -5.5841e+01,
        -6.7208e+01, -3.8671e+01, -4.7897e+01, -3.9218e+01, -3.5496e+01,
        -2.4081e+01, -3.4612e+01, -5.6449e+01, -6.1959e+01, -4.4326e+01,
        -4.4054e+01, -5.2762e+01, -5.5077e+01, -5.8257e+01, -3.4381e+01,
        -4.4991e+01, -3.6387e+01, -4.8368e+01, -4.6085e+01, -5.9845e+01,
        -4.7140e+01, -5.5151e+01, -4.6223e+01, -4.7704e+01, -4.0940e+01,
        -7.6307e+01, -5.7265e+01, -5.9486e+01, -4.4185e+01, -4.7157e+01,
        -3.2520e+01, -4.4903e+01, -4.1965e+01, -3.8224e+01, -4.1127e+01,
        -3.9249e+01, -2.1862e+01, -3.7908e+01, -6.2629e+01, -4.4977e+01,
        -4.1157e+01, -6.3431e+01, -5.6249e+01, -5.7644e+01, -5.5751e+01,
        -2.3353e+01, -5.5436e+01, -2.2254e+01, -5.7826e+01, -4.8440e+01,
        -5.5493e+01, -2.5767e+01, -2.7604e+01, -5.5374e+01, -2.6229e+01,
        -3.9994e+01, -4.0397e+01, -4.8714e+01, -3.7475e+01, -3.4994e+01,
        -5.2787e+01, -3.7054e+01, -3.3569e+01, -3.7697e+01, -1.8695e+01,
        -3.1738e+01, -5.1994e+01, -5.2801e+01,  1.2098e+01, -2.3888e+01,
        -9.5789e+00, -2.9090e+01], device='cuda:0')
loss:    tensor(8.9978, device='cuda:0')
"""
