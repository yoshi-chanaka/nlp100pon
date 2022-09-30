from mask import generate_square_subsequent_mask
from convert_sequence import convert_sent2idx
from translate import load_model
from process import post_process
from load import load_data

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import TensorDataset, DataLoader
import sacrebleu
from tqdm import tqdm


def greedy_decode_corpus(model, src, src_mask, src_padding_mask, max_lens, start_symbol, device, EOS_IDX):

    src = src.to(device)
    src_mask = src_mask.to(device)
    src_padding_mask = src_padding_mask.to(device)

    memories = model.transformer.encoder(
        model.positional_encoding(model.src_tok_emb(
            src)), src_mask, src_padding_mask
    ).transpose(0, 1)
    output_list = []
    for idx, memory in enumerate(memories):

        ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)
        memory = memory.unsqueeze(0).transpose(0, 1)[:max_lens[idx]].to(device)

        for i in range(max_lens[idx] + 9):

            tgt_mask = (generate_square_subsequent_mask(
                ys.size(0)).type(torch.bool)).to(device)
            out = model.decode(ys, memory, tgt_mask)
            out = out.transpose(0, 1)
            prob = model.generator(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            next_word = next_word.item()

            ys = torch.cat([ys,
                            torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
            if next_word == EOS_IDX:
                break
        output_list.append(ys.flatten())

    return output_list


def translate_corpus(
    model: torch.nn.Module,
    src_corpus,
    vocab_src,
    vocab_tgt,
    device,
    PAD_IDX=1,
    BOS_IDX=2,
    EOS_IDX=3,
    post_proc=False,
    batch_size=16
):

    src_indices, src_lengths = [], []
    for src_sentence in src_corpus:
        ids = convert_sent2idx(src_sentence.split(
            ' '), vocab_src, set(list(vocab_src.keys())))
        src_indices.append(torch.Tensor(ids).int())
        src_lengths.append(len(ids))

    src = pad_sequence(src_indices, batch_first=True,
                        padding_value=PAD_IDX).long()
    src_lengths = torch.Tensor(src_lengths).long()
    dataloader = DataLoader(TensorDataset(
        src, src_lengths), batch_size=batch_size, shuffle=False)

    output_list = []
    model.eval()
    for src, length in tqdm(dataloader):

        src = src.transpose(0, 1).to(device)
        num_tokens = src.shape[0]
        src_mask = (torch.zeros(num_tokens, num_tokens)
                    ).type(torch.bool).to(device)
        src_padding_mask = (src == PAD_IDX).transpose(0, 1)

        tgt_tokens_list = greedy_decode_corpus(
            model, src, src_mask, src_padding_mask,
            max_lens=length, start_symbol=BOS_IDX, device=device, EOS_IDX=EOS_IDX
        )

        vocab_decode = list(vocab_tgt.keys())
        for tgt_tokens in tgt_tokens_list:
            output = " ".join([vocab_decode[idx.item()] for idx in tgt_tokens])
            if post_proc:
                output_list.append(post_process(
                    output.replace("<bos>", "").replace("<eos>", "")))
            else:
                output_list.append(output.replace(
                    "<bos>", "").replace("<eos>", ""))

    return output_list


def compute_corpus_bleu(model, srcpath, tgtpath, device,
                vocab_src, vocab_tgt, batch_size=16, num_sentences=None):

    with open(srcpath, 'r', encoding='utf-8') as f:
        if num_sentences == None:
            num_sentences = len(f.readlines())
        else:
            num_sentences = min(num_sentences, len(f.readlines()))

    src_corpus = []
    pred_corpus, true_corpus = [], []
    f_src = open(srcpath, 'r', encoding='utf-8')
    f_tgt = open(tgtpath, 'r', encoding='utf-8')

    line_src = f_src.readline()
    line_tgt = f_tgt.readline()

    for i in range(num_sentences):

        """
        in: '▁ 道 元 ( どう げん ) は 、 鎌倉時代 初期の 禅 僧 。'
        out: '▁Do gen ▁was ▁a ▁Zen ▁monk ▁in ▁the ▁early ▁Kamakura ▁period .'
        """
        src_corpus.append(line_src.strip())
        true_corpus.append(post_process(line_tgt.strip()))

        line_src = f_src.readline()
        line_tgt = f_tgt.readline()

    f_src.close()
    f_tgt.close()

    with torch.no_grad():
        pred_corpus = translate_corpus(
                model=model,
                src_corpus=src_corpus,
                vocab_src=vocab_src,
                vocab_tgt=vocab_tgt,
                device=device,
                post_proc=True,
                batch_size=batch_size
        )
    bleu = sacrebleu.corpus_bleu(pred_corpus, [true_corpus])

    return bleu


if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    _, vocab_ja, vocab_en = load_data(only_vocab=True)
    model = load_model(device=device)

    for k in ['train', 'dev', 'test']:
        srcpath = f'../data/{k}_ja.kftt'
        tgtpath = f'../data/{k}_en.kftt'

        bleu = compute_corpus_bleu(model, srcpath, tgtpath,
                            device, vocab_ja, vocab_en,
                            num_sentences=10000,
                            batch_size=128)
        print(f'{k}\t{bleu}') # bleu.score
"""
Done
100%|███████████████████████████████████████████████████████████████████████████████████| 79/79 [18:20<00:00, 13.93s/it]
train   BLEU = 23.87 57.2/31.3/19.7/13.1 (BP = 0.915 ratio = 0.919 hyp_len = 247854 ref_len = 269739)
100%|███████████████████████████████████████████████████████████████████████████████████| 10/10 [01:50<00:00, 11.02s/it]
dev     BLEU = 18.04 49.5/23.2/12.6/7.3 (BP = 1.000 ratio = 1.016 hyp_len = 24662 ref_len = 24281)
100%|███████████████████████████████████████████████████████████████████████████████████| 10/10 [01:58<00:00, 11.90s/it]
test    BLEU = 20.53 51.4/25.4/14.9/9.5 (BP = 0.990 ratio = 0.990 hyp_len = 26296 ref_len = 26563)
"""
