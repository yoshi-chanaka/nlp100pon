from load import load_data
from translate import load_model, translate, post_process

import sacrebleu
import torch
from tqdm import tqdm


def compute_bleu(model, srcpath, tgtpath, device, num_sentences=None):
    
    with open(srcpath, 'r', encoding='utf-8') as f:
        if num_sentences == None:
            num_sentences = len(f.readlines())
        else:
            num_sentences = min(num_sentences,len(f.readlines()))
        
    pred_corpus, true_corpus = [], []
    f_src = open(srcpath, 'r', encoding='utf-8')
    f_tgt = open(tgtpath, 'r', encoding='utf-8')

    line_src = f_src.readline()
    line_tgt = f_tgt.readline()

    for i in tqdm(range(num_sentences)):
        
        """
        in: '▁ 道 元 ( どう げん ) は 、 鎌倉時代 初期の 禅 僧 。'
        out: '▁Do gen ▁was ▁a ▁Zen ▁monk ▁in ▁the ▁early ▁Kamakura ▁period .'
        """
        pred = translate(
            model=model, 
            src_sentence=line_src.strip(), 
            vocab_src=vocab_ja, 
            vocab_tgt=vocab_en, 
            device=device, 
            post_proc=False
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

        bleu = compute_bleu(model, srcpath, tgtpath, device, num_sentences=10000)
        print(f'{k}\t{bleu}')
    

"""
Using cuda device
Loading Dataset ...
Done
Loading Model ... : Using cuda device
Done
100%|███████████████████████████████████████████████████████████| 10000/10000 [20:28<00:00,  8.14it/s]
train   BLEU = 22.23 58.0/31.7/19.9/13.3 (BP = 0.841 ratio = 0.853 hyp_len = 230030 ref_len = 269739)
100%|█████████████████████████████████████████████████████████████| 1166/1166 [01:59<00:00,  9.80it/s]
dev     BLEU = 17.50 50.7/23.8/13.0/7.5 (BP = 0.943 ratio = 0.944 hyp_len = 22933 ref_len = 24281)
100%|█████████████████████████████████████████████████████████████| 1160/1160 [02:05<00:00,  9.23it/s]
test    BLEU = 19.40 52.3/25.9/15.1/9.7 (BP = 0.920 ratio = 0.923 hyp_len = 24517 ref_len = 26563)
"""
