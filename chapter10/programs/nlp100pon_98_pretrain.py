from model import TransformerNMTModel
from mask import create_mask
from load import load_data
from nlp100pon_93_bleu import compute_bleu
from evaluate import compute_corpus_bleu

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
import os
import copy
import datetime
from torch.utils.tensorboard import SummaryWriter

"""
nvidia-smi
nohup python nlp100pon_98_pretrain.py > ../materials/output_pretrain.txt &
"""

if __name__ == "__main__":

    print(datetime.datetime.now())

    batch_size = 128
    dataset, vocab_src, vocab_tgt = load_data(extension='jpc')
    print('Train Size:\t', dataset['train'][:][0].shape, dataset['train'][:][1].shape)
    print('Dev Size:\t', dataset['dev'][:][0].shape, dataset['dev'][:][1].shape)
    print('Test Size:\t', dataset['test'][:][0].shape, dataset['test'][:][1].shape)
    train_dataloader = DataLoader(
        dataset['train'], batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(
        dataset['dev'], batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(
        dataset['test'], batch_size=batch_size, shuffle=False)

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
    # bleuを測るためのデータファイルの指定
    bleu_srcpath = f'../data/dev_ja.jpc'
    bleu_tgtpath = f'../data/dev_en.jpc'
    tmp_modelpath = f'tmp.pth'

    model = TransformerNMTModel(
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        emb_size=EMB_SIZE,
        src_vocab_size=SRC_VOCAB_SIZE,
        tgt_vocab_size=TGT_VOCAB_SIZE,
        nhead=NHEAD,
        dim_feedforward=FFN_HID_DIM,
    )
    model_orig = copy.deepcopy(model)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    model = model.to(device)
    model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])

    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = optim.Adam(model.parameters(), lr=0.0001,
                           betas=(0.9, 0.98), eps=1e-9)

    model_dir_path = Path('../models')
    if not model_dir_path.exists():
        model_dir_path.mkdir(parents=True)

    writer = SummaryWriter(log_dir="../materials/logs_pretrained")
    NUM_EPOCHS = 30
    for epoch in range(NUM_EPOCHS):

        model.train()
        losses = 0
        cnt = 0
        for src, tgt in tqdm(train_dataloader):
            # (バッチサイズ, 系列長) -> (系列長, バッチサイズ)
            src = src.transpose(0, 1).to(device)
            tgt = tgt.transpose(0, 1).long().to(device)

            tgt_input, tgt_output = tgt[:-1, :], tgt[1:, :]
            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(
                src, tgt_input)
            src_mask, tgt_mask = src_mask.to(device), tgt_mask.to(device)

            # (系列長, バッチサイズ) -> (バッチサイズ, 系列長)
            src_tr, tgt_input_tr = src.transpose(
                0, 1), tgt_input.transpose(0, 1)
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
            loss = loss_fn(
                pred.reshape(-1, pred.shape[-1]), tgt_output.reshape(-1))
            loss.backward()
            optimizer.step()
            losses += loss.item()
            cnt += 1
            if cnt % 100 == 0:
                del pred
                torch.cuda.empty_cache()

        train_loss = (losses / len(train_dataloader))

        model.eval()
        losses = 0
        cnt = 0
        for src, tgt in valid_dataloader:
            src = src.transpose(0, 1).to(device)
            tgt = tgt.transpose(0, 1).long().to(device)

            tgt_input, tgt_output = tgt[:-1, :], tgt[1:, :]
            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(
                src, tgt_input)
            src_mask, tgt_mask = src_mask.to(device), tgt_mask.to(device)

            src_tr, tgt_input_tr = src.transpose(
                0, 1), tgt_input.transpose(0, 1)
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

            loss = loss_fn(
                pred.reshape(-1, pred.shape[-1]), tgt_output.reshape(-1))
            losses += loss.item()
            cnt += 1
            if cnt % 100 == 0:
                del pred
                torch.cuda.empty_cache()

        valid_loss = (losses / len(valid_dataloader))

        torch.save(model.module.state_dict(), tmp_modelpath)
        eval_model = copy.deepcopy(model_orig)
        eval_model = eval_model.to(device)
        eval_model.load_state_dict(torch.load(tmp_modelpath, map_location=device))

        eval_model.eval()
        with torch.no_grad():
            valid_bleu = compute_corpus_bleu(eval_model, bleu_srcpath, bleu_tgtpath,
                                    device, vocab_src, vocab_tgt, num_sentences=None, 
                                    batch_size=32)
        os.remove(tmp_modelpath)
        del eval_model
        torch.cuda.empty_cache()

        print('Epoch: {} Train loss: {:.4f}, Valid loss: {:.4f}, Valid BLEU: {:.4f}'.
              format(epoch + 1, train_loss, valid_loss, valid_bleu.score))
        writer.add_scalars('data/loss',
                    {
                        'train': train_loss,
                        'dev': valid_loss},
                    (epoch))
        writer.add_scalars('data/bleu',
                    {
                        'bleu': valid_bleu.score},
                    (epoch))
        if (epoch + 1) % 5 == 0:
            torch.save(model.module.state_dict(),
            model_dir_path.joinpath(f'model_pretrained_{str(epoch + 1).zfill(2)}epochs.pth'))

    writer.close()
    torch.save(model.module.state_dict(), model_dir_path.joinpath('model_pretrained.pth'))
    print(datetime.datetime.now())
