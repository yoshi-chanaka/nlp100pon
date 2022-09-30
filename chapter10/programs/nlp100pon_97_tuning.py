from model import TransformerNMTModel
from mask import create_mask
from load import load_data
from translate import load_model, translate
from process import post_process

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from pathlib import Path
import optuna
import pickle
import os
import copy
import datetime
import sacrebleu

"""
nvidia-smi
nohup python -u nlp100pon_97_tuning.py >& ../materials/output_tune.txt &
"""
def compute_bleu(model, srcpath, tgtpath, device,
                vocab_src, vocab_tgt, num_sentences=None):

    with open(srcpath, 'r', encoding='utf-8') as f:
        if num_sentences == None:
            num_sentences = len(f.readlines())
        else:
            num_sentences = min(num_sentences, len(f.readlines()))

    pred_corpus, true_corpus = [], []
    f_src = open(srcpath, 'r', encoding='utf-8')
    f_tgt = open(tgtpath, 'r', encoding='utf-8')

    line_src = f_src.readline()
    line_tgt = f_tgt.readline()

    model.eval()
    for i in range(num_sentences):

        """
        in: '▁ 道 元 ( どう げん ) は 、 鎌倉時代 初期の 禅 僧 。'
        out: '▁Do gen ▁was ▁a ▁Zen ▁monk ▁in ▁the ▁early ▁Kamakura ▁period .'
        """
        pred = translate(
            model=model,
            src_sentence=line_src.strip(),
            vocab_src=vocab_src,
            vocab_tgt=vocab_tgt,
            device=device,
            post_proc=False,
            margin=50
        )

        pred_corpus.append(post_process(pred))
        true_corpus.append(post_process(line_tgt.strip()))
        # pred_corpus.append(pred)
        # true_corpus.append(line_tgt.strip())

        line_src = f_src.readline()
        line_tgt = f_tgt.readline()

    f_src.close()
    f_tgt.close()

    bleu = sacrebleu.corpus_bleu(pred_corpus, [true_corpus])

    return bleu

def objective(trial):

    global train_dataloader, valid_dataloader, vocab_src, vocab_tgt, device

    torch.manual_seed(0)

    SRC_VOCAB_SIZE = len(vocab_src)
    TGT_VOCAB_SIZE = len(vocab_tgt)

    EMB_SIZE =              int(trial.suggest_int('emb_size', 128, 1024, 64))
    NHEAD =                 int(trial.suggest_int('nhead', 8, 16, 8))
    FFN_HID_DIM =           int(trial.suggest_int('ffn_hid_dim', 128, 1024, 64))
    NUM_ENCODER_LAYERS =    int(trial.suggest_int('num_encoder_layers', 1, 4, 1))
    NUM_DECODER_LAYERS =    int(trial.suggest_int('num_decoder_layers', 1, 4, 1))

    pad_idx = 1
    bleu_srcpath = f'../data/dev_ja.kftt'
    bleu_tgtpath = f'../data/dev_en.kftt'
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
    optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

    valid_bleu_history = []
    NUM_EPOCHS = 30
    for epoch in range(NUM_EPOCHS):

        model.train()
        losses = 0
        cnt = 0
        for src, tgt in train_dataloader:
            # (バッチサイズ, 系列長) -> (系列長, バッチサイズ)
            src = src.transpose(0, 1).to(device)
            tgt = tgt.transpose(0, 1).long()

            tgt_input, tgt_output = tgt[:-1, :].to(device), tgt[1:, :].to(device)
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

        # 検証用データに対するBLEUの計測
        torch.save(model.module.state_dict(), tmp_modelpath)
        eval_model = copy.deepcopy(model_orig)
        eval_model = eval_model.to(device)
        eval_model.load_state_dict(torch.load(tmp_modelpath, map_location=device))

        eval_model.eval()
        with torch.no_grad():
            valid_bleu = compute_bleu(eval_model, bleu_srcpath, bleu_tgtpath,
                                    device, vocab_src, vocab_tgt, num_sentences=None)
        valid_bleu_history.append(valid_bleu.score)

        print('{}\tEpoch: {} Train [loss: {:.4f}], Valid [BLEU: {:.4f}]'.
                format(datetime.datetime.now(), epoch + 1, losses / len(train_dataloader), valid_bleu.score))
        os.remove(tmp_modelpath)
        del eval_model
        torch.cuda.empty_cache()

    del model, model_orig
    torch.cuda.empty_cache()

    return max(valid_bleu_history)

if __name__ == "__main__":

    train_size = 200000
    batch_size = 128
    dataset, vocab_src, vocab_tgt = load_data()
    train_src, train_tgt = dataset['train'][:train_size]

    train_dataloader    = DataLoader(TensorDataset(train_src, train_tgt),
                                        batch_size=batch_size, shuffle=True)
    valid_dataloader    = DataLoader(dataset['dev'], batch_size=batch_size, shuffle=False)
    test_dataloader     = DataLoader(dataset['test'], batch_size=batch_size, shuffle=False)

    print(datetime.datetime.now())
    print('batch size: {}'.format(batch_size))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} device'.format(device))

    TRIAL_SIZE = 20
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=TRIAL_SIZE)

    print('best parameters: \n{}'.format(study.best_params))
    print('best value: {}'.format(study.best_value))

    best_result = {
        'best_parameters': study.best_params,
        'best_value': study.best_value
    }
    savepath = '../models/tuning_best_params.pickle'
    with open(savepath, 'wb') as f:
        pickle.dump(best_result, f)

    print(datetime.datetime.now())
