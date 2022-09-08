from model import TransformerNMTModel
from mask import generate_square_subsequent_mask
from convert_sequence import convert_sent2idx

import torch

def load_model(
    path = '../models/model.pth',
    src_vocab_size = 8000,
    tgt_vocab_size = 8000,
    emb_size = 512,
    nhead = 8,
    ffn_hid_dim = 512,
    num_encoder_layers = 3,
    num_decoder_layers = 3,
    model = None,
    device='cuda'
):
    """
    読み込むモデルの構造と同じ構造のモデルを用意して
    model.load_state_dict(torch.load(path, map_location=device)) で呼び出す
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Loading Model ... : Using {} device'.format(device))

    if model == None:

        model = TransformerNMTModel(
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            emb_size=emb_size,
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            nhead=nhead,
            dim_feedforward=ffn_hid_dim,
        )

    model = model.to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    print('Done')
    
    return model

def post_process(input_text):

    tokens = input_text.strip().split()
    out_tokens = []
    for i, tok in enumerate(tokens):
        if tok[0] == '▁':
            out_tokens.append(tok[1:])
        elif i == 0:
            out_tokens.append(tok)
        else:
            out_tokens[-1] += tok
    return ' '.join(out_tokens)

# function to generate output sequence using greedy algorithm
def greedy_decode(model, src, src_mask, max_len, start_symbol, device, EOS_IDX):

    src = src.to(device)
    src_mask = src_mask.to(device)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)

    for i in range(max_len - 1):

        memory = memory.to(device)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0)).type(torch.bool)).to(device)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == EOS_IDX:
            break

    return ys


# actual function to translate input sentence into target language
def translate(
    model: torch.nn.Module, 
    src_sentence: str,
    vocab_src,
    vocab_tgt,
    device, 
    BOS_IDX = 2, 
    EOS_IDX = 3,
    post_proc=False
):
    src = torch.Tensor(convert_sent2idx(src_sentence.split(' '), vocab_src, set(list(vocab_src.keys())))).long()
    src = src.unsqueeze(0).reshape(-1, 1)
    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)

    model.eval()
    tgt_tokens = greedy_decode(
        model, src, src_mask, 
        max_len=num_tokens + 5, start_symbol=BOS_IDX, device=device, EOS_IDX=EOS_IDX
    ).flatten()
    vocab_decode = list(vocab_tgt.keys())
    output = " ".join([vocab_decode[idx.item()] for idx in tgt_tokens])
    if post_proc:
        return post_process(output.replace("<bos>", "").replace("<eos>", ""))
    else:
        return output.replace("<bos>", "").replace("<eos>", "")



if __name__ == "__main__":
    
    from load import load_data
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using {} device'.format(device))

    _, vocab_ja, vocab_en = load_data(only_vocab=True)
    model = load_model(device=device)

    # train
    text = "▁ 没 年 は 、 確 実 な 記録 はない が 1 50 6 年 とする ものが多い 。"
    print('\n' + text)
    print(translate(model=model, src_sentence=text, vocab_src=vocab_ja, vocab_tgt=vocab_en, device=device, post_proc=True))
    # ▁There ▁is ▁no ▁reliable ▁record ▁of ▁the ▁date ▁of ▁his ▁death , ▁but ▁most ▁put ▁it ▁at ▁15 06 .

    # dev
    text = "▁ 一般に 禅宗 は 知識 ではなく 、 悟 り を 重 んじ る 。"
    print('\n' + text)
    print(translate(model=model, src_sentence=text, vocab_src=vocab_ja, vocab_tgt=vocab_en, device=device, post_proc=True))
    # ▁The ▁Zen ▁sects ▁generally ▁emphasize ▁enlightenment ▁over ▁knowledge .
    
    # test
    text = "▁ 特 記事 項 の ない ものは 19 7 2 年に 前 所属 機関 区 から 転 入 の手 続き が と られている 。"
    print('\n' + text)
    print(translate(model=model, src_sentence=text, vocab_src=vocab_ja, vocab_tgt=vocab_en, device=device, post_proc=True))
    # ▁Un less ▁other wise ▁not ed , ▁the ▁steam ▁locomotive s ▁were ▁transferred ▁from ▁the ▁previous ▁en gi ne ▁de po ts ▁in ▁1972 .


"""
Using cuda device
Loading Dataset ...
Done
Loading Model ... : Using cuda device
Done

▁ 没 年 は 、 確 実 な 記録 はない が 1 50 6 年 とする ものが多い 。
Although there are no certain records about his death, most of them say that he died in 1506.
DeepL: 彼の死については確かな記録はないが、1506年に亡くなったとするものが多い。

▁ 一般に 禅宗 は 知識 ではなく 、 悟 り を 重 んじ る 。
Generally, Zen sect is not knowledge but emphasizes enlightenment.
DeepL: 一般に禅宗は知識ではなく、悟りに重きを置いている。

▁ 特 記事 項 の ない ものは 19 7 2 年に 前 所属 機関 区 から 転 入 の手 続き が と られている 。
In 1972, the procedures of transferring from the former institution to the institution belong to the prefecture are considered to be procedures.
DeepL: 1972年、旧施設から都道府県に属する施設へ移行する手続きとされる。
"""
