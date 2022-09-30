from load import load_data
from translate import load_model, translate
from process import post_process

import sacrebleu
import torch
from tqdm import tqdm


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


if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using {} device'.format(device))

    _, vocab_ja, vocab_en = load_data(only_vocab=True)
    model = load_model(device=device)

    for k in ['train', 'dev', 'test']:
        srcpath = f'../data/{k}_ja.kftt'
        tgtpath = f'../data/{k}_en.kftt'

        bleu = compute_bleu(model, srcpath, tgtpath, device, 
                            vocab_ja, vocab_en, num_sentences=10000)
        print(f'{k}\t{bleu}')


"""
Using cuda device
Loading Dataset ...
Done
Loading Model ... : Using cuda device
Done
train   BLEU = 24.32 53.7/29.2/18.3/12.2 (BP = 1.000 ratio = 1.060 hyp_len = 285834 ref_len = 269739)
dev     BLEU = 16.50 45.5/21.2/11.5/6.7 (BP = 1.000 ratio = 1.178 hyp_len = 28606 ref_len = 24281)
test    BLEU = 18.76 47.2/23.1/13.4/8.5 (BP = 1.000 ratio = 1.139 hyp_len = 30250 ref_len = 26563)
"""
