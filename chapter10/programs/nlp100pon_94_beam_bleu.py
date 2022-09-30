from load import load_data
from translate import load_model, translate
from process import post_process

import sacrebleu
import torch
from tqdm import tqdm
from timeit import default_timer as timer

"""
python nlp100pon_94_beam_bleu.py > ../materials/beam_search.txt
"""

def beam_compute_bleu(model, srcpath, tgtpath, device,
                vocab_src, vocab_tgt, num_sentences=None, beam_width=2):

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
    for i in tqdm(range(num_sentences)):

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
            method='beam',
            beam_width=beam_width,
            margin=50
        )

        pred_corpus.append(post_process(pred))
        true_corpus.append(post_process(line_tgt.strip()))

        line_src = f_src.readline()
        line_tgt = f_tgt.readline()

    f_src.close()
    f_tgt.close()

    bleu = sacrebleu.corpus_bleu(pred_corpus, [true_corpus])

    return bleu


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using {} device'.format(device))

    _, vocab_ja, vocab_en = load_data(only_vocab=True)
    model = load_model(device=device)

    beam_width_list = [2 ** i for i in range(6)]

    srcpath = f'../data/test_ja.kftt'
    tgtpath = f'../data/test_en.kftt'

    bleu_scores = []
    times = []
    for beam_width in beam_width_list:

        start_time = timer()
        bleu = beam_compute_bleu(model, srcpath, tgtpath, device,
                            vocab_ja, vocab_en, num_sentences=None,
                            beam_width=beam_width)
        end_time = timer()
        print(f'beam_with: {beam_width}\t{bleu}')
        bleu_scores.append(bleu.score)
        times.append(end_time - start_time)

    fig, ax = plt.subplots()
    ax.plot(beam_width_list, bleu_scores)
    ax.set_xlabel('beam width')
    ax.set_ylabel('BLEU')
    ax.set_xscale('log', basex=2)
    plt.legend()
    plt.savefig('../figures/beam_search_bleu.jpg')

    fig, ax = plt.subplots()
    plt.plot(beam_width_list, times)
    ax.set_xlabel('beam width')
    ax.set_ylabel('time')
    ax.set_xscale('log', basex=2)
    plt.legend()
    plt.savefig('../figures/beam_search_time.jpg')

