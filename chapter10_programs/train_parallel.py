from model import TransformerNMTModel
from mask import create_mask
from load import load_data

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from timeit import default_timer as timer
from tqdm import tqdm
from pathlib import Path

if __name__ == "__main__":

    batch_size = 128
    dataset, vocab_src, vocab_tgt = load_data()
    train_dataloader    = DataLoader(dataset['train'], batch_size=batch_size, shuffle=True)
    valid_dataloader    = DataLoader(dataset['dev'], batch_size=batch_size, shuffle=False)
    test_dataloader     = DataLoader(dataset['test'], batch_size=batch_size, shuffle=False)

    print('batch size: {}'.format(batch_size))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using {} device'.format(device))

    torch.manual_seed(0)

    SRC_VOCAB_SIZE = len(vocab_src)
    TGT_VOCAB_SIZE = len(vocab_tgt)
    EMB_SIZE = 512
    NHEAD = 8
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

    model = model.to(device)
    model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])
    
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    
    f = open('../materials/train_parallel.csv', 'w')
    model_dir_path = Path('../models')
    if not model_dir_path.exists():
        model_dir_path.mkdir(parents=True)

    NUM_EPOCHS = 30
    for epoch in range(NUM_EPOCHS):

        start_time = timer()
        model.train()
        losses = 0
        cnt = 0
        for src, tgt in tqdm(train_dataloader):
            # (バッチサイズ, 系列長) -> (系列長, バッチサイズ)
            src = src.transpose(0, 1).to(device)
            tgt = tgt.transpose(0, 1).long().to(device)

            tgt_input, tgt_output = tgt[:-1, :], tgt[1:, :]
            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)
            src_mask, tgt_mask = src_mask.to(device), tgt_mask.to(device)

            # (系列長, バッチサイズ) -> (バッチサイズ, 系列長)
            src_tr, tgt_input_tr = src.transpose(0, 1), tgt_input.transpose(0, 1)
            pred = model(
                src=src_tr,
                trg=tgt_input_tr,
                src_mask=src_mask.repeat(src_tr.shape[0], 1, 1),
                tgt_mask=tgt_mask.repeat(src_tr.shape[0], 1, 1),
                src_padding_mask=src_padding_mask,
                tgt_padding_mask=tgt_padding_mask,
                memory_key_padding_mask=src_padding_mask
            )
            pred = pred.transpose(0, 1)

            optimizer.zero_grad()
            loss = loss_fn(pred.reshape(-1, pred.shape[-1]), tgt_output.reshape(-1))
            loss.backward()
            optimizer.step()
            losses += loss.item()
            cnt += 1
            if cnt % 100 == 0:
                del pred
                torch.cuda.empty_cache()

        train_loss =  (losses / len(train_dataloader))

        model.eval()
        losses = 0
        cnt = 0
        for src, tgt in valid_dataloader:
            src = src.transpose(0, 1).to(device)
            tgt = tgt.transpose(0, 1).long().to(device)

            tgt_input, tgt_output = tgt[:-1, :], tgt[1:, :]
            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)
            src_mask, tgt_mask = src_mask.to(device), tgt_mask.to(device)

            src_tr, tgt_input_tr = src.transpose(0, 1), tgt_input.transpose(0, 1)
            with torch.no_grad():
                pred = model(
                    src=src_tr,
                    trg=tgt_input_tr,
                    src_mask=src_mask.repeat(src_tr.shape[0], 1, 1),
                    tgt_mask=tgt_mask.repeat(src_tr.shape[0], 1, 1),
                    src_padding_mask=src_padding_mask,
                    tgt_padding_mask=tgt_padding_mask,
                    memory_key_padding_mask=src_padding_mask
                )
            pred = pred.transpose(0, 1)

            loss = loss_fn(pred.reshape(-1, pred.shape[-1]), tgt_output.reshape(-1))
            losses += loss.item()
            cnt += 1
            if cnt % 100 == 0:
                del pred
                torch.cuda.empty_cache()

        valid_loss = (losses / len(valid_dataloader))
        end_time = timer()

        """
        BLEUも表示したい
        """
        print('Epoch: {} Train loss: {:.4f}, Valid loss: {:.4f}, Epoch time: {:.4f}'.
                format(epoch + 1, train_loss, valid_loss, end_time - start_time))
        f.write(f'{epoch + 1},{train_loss},{valid_loss},{end_time - start_time}\n')
        if (epoch + 1) % 5 == 0:
            torch.save(model.module.state_dict(), model_dir_path.joinpath(f'model_{str(epoch + 1).zfill(2)}epochs.pth'))

    f.close()
    torch.save(model.module.state_dict(), model_dir_path.joinpath('model.pth'))
    
