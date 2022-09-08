from model import TransformerNMTModel
from mask import create_mask
from load import load_data

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from timeit import default_timer as timer
from tqdm import tqdm

if __name__ == "__main__":

    batch_size = 8
    dataset, vocab_src, vocab_tgt = load_data()
    train_dataloader    = DataLoader(dataset['train'], batch_size=batch_size, shuffle=True)
    valid_dataloader    = DataLoader(dataset['dev'], batch_size=batch_size, shuffle=False)
    test_dataloader     = DataLoader(dataset['test'], batch_size=batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(0)

    SRC_VOCAB_SIZE = len(vocab_src)
    TGT_VOCAB_SIZE = len(vocab_tgt)
    EMB_SIZE = 512
    NHEAD = 64
    FFN_HID_DIM = 512
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

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using {} device'.format(device))

    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_idx)

    optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

    NUM_EPOCHS = 20
    for epoch in range(NUM_EPOCHS):

        start_time = timer()
        model.train()
        losses = 0
        for src, tgt in tqdm(train_dataloader):
            # (系列長, データ数)
            src = src.transpose(0, 1).to(device)
            tgt = tgt.transpose(0, 1).long().to(device)

            tgt_input, tgt_output = tgt[:-1, :], tgt[1:, :]
            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)
            src_mask, tgt_mask = src_mask.to(device), tgt_mask.to(device)
            pred = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)

            optimizer.zero_grad()
            loss = loss_fn(pred.reshape(-1, pred.shape[-1]), tgt_output.reshape(-1))
            loss.backward()
            optimizer.step()
            losses += loss.item()

        train_loss =  losses / len(train_dataloader)
        end_time = timer()


        model.eval()
        losses = 0
        for src, tgt in valid_dataloader:
            src = src.transpose(0, 1).to(device)
            tgt = tgt.transpose(0, 1).long().to(device)

            tgt_input, tgt_output = tgt[:-1, :], tgt[1:, :]
            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)
            src_mask, tgt_mask = src_mask.to(device), tgt_mask.to(device)
            pred = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)

            loss = loss_fn(pred.reshape(-1, pred.shape[-1]), tgt_output.reshape(-1))
            losses += loss.item()

        valid_loss = losses / len(valid_dataloader)

        """
        BLEUも表示したい
        """
        print('Epoch: {} Train [loss: {:.4f}], Valid [loss: {:.4f}], Epoch time: {:.4f}'.
                format(epoch + 1, train_loss, valid_loss, end_time - start_time))
