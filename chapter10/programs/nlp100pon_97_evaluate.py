from load import load_data
from translate import load_model, translate
from process import post_process
from nlp100pon_97_tuning import compute_bleu

import torch

if __name__ == "__main__":

    import sentencepiece as spm
    import pickle

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using {} device'.format(device))

    _, vocab_ja, vocab_en = load_data(only_vocab=True)

    hyparams_path = '../models/tuning_best_params.pickle'
    with open(hyparams_path, 'rb') as f:
        tuned_dict = pickle.load(f)
    tuned_params = tuned_dict['best_parameters']

    SRC_VOCAB_SIZE = len(vocab_ja)
    TGT_VOCAB_SIZE = len(vocab_en)
    EMB_SIZE = tuned_params['emb_size']
    NHEAD = tuned_params['nhead']
    FFN_HID_DIM = tuned_params['ffn_hid_dim']
    NUM_ENCODER_LAYERS = tuned_params['num_encoder_layers']
    NUM_DECODER_LAYERS = tuned_params['num_decoder_layers']

    modelpath = '../models/model_tuned.pth'
    model = load_model(
        path = modelpath,
        src_vocab_size = SRC_VOCAB_SIZE,
        tgt_vocab_size = TGT_VOCAB_SIZE,
        emb_size = EMB_SIZE,
        nhead = NHEAD,
        ffn_hid_dim = FFN_HID_DIM,
        num_encoder_layers = NUM_ENCODER_LAYERS,
        num_decoder_layers = NUM_DECODER_LAYERS,
        model = None,
        device=device
    )

    sp_tokenizer = spm.SentencePieceProcessor(model_file='../models/kftt_sp_ja.model')

    input_text = "今日は良い天気ですね。"
    proc_text = ' '.join(sp_tokenizer.EncodeAsPieces(input_text))
    print('\n' + proc_text)
    print(translate(model=model, src_sentence=proc_text, vocab_src=vocab_ja, vocab_tgt=vocab_en, device=device, post_proc=True))
    # It is good at present.

    input_text = "私の考える良い研究テーマの決め方について書きます。"
    proc_text = ' '.join(sp_tokenizer.EncodeAsPieces(input_text))
    print('\n' + proc_text)
    print(translate(model=model, src_sentence=proc_text, vocab_src=vocab_ja, vocab_tgt=vocab_en, device=device, post_proc=True))
    # I will write on how theme of themes of themes of the good research, which we considers me.

    input_text = "明日から再び研究室の活動が始まります。"
    proc_text = ' '.join(sp_tokenizer.EncodeAsPieces(input_text))
    print('\n' + proc_text)
    print(translate(model=model, src_sentence=proc_text, vocab_src=vocab_ja, vocab_tgt=vocab_en, device=device, post_proc=True))
    # Starting with the activity of the research room from the Ming Dynasty, the study room started again.

    # train
    text = "▁ 没 年 は 、 確 実 な 記録 はない が 1 50 6 年 とする ものが多い 。"
    print('\n' + text)
    print(translate(model=model, src_sentence=text, vocab_src=vocab_ja, vocab_tgt=vocab_en, device=device, post_proc=True))
    # There is no reliable record of his death, but many of them are said to have been 1506.

    # dev
    text = "▁ 一般に 禅宗 は 知識 ではなく 、 悟 り を 重 んじ る 。"
    print('\n' + text)
    print(translate(model=model, src_sentence=text, vocab_src=vocab_ja, vocab_tgt=vocab_en, device=device, post_proc=True))
    # In general, Zen is not knowledge but emphasized enlightenment.

    # test
    text = "▁ 特 記事 項 の ない ものは 19 7 2 年に 前 所属 機関 区 から 転 入 の手 続き が と られている 。"
    print('\n' + text)
    print(translate(model=model, src_sentence=text, vocab_src=vocab_ja, vocab_tgt=vocab_en, device=device, post_proc=True))
    # In 1972, the procedure for transfer from the former institutional institutions without special articles was established.

    for k in ['train', 'dev', 'test']:
        srcpath = f'../data/{k}_ja.kftt'
        tgtpath = f'../data/{k}_en.kftt'

        bleu = compute_bleu(model, srcpath, tgtpath, device, 
                            vocab_ja, vocab_en, num_sentences=10000)
        print(f'{k}\t{bleu}')


"""
train   BLEU = 24.56 52.6/29.3/18.7/12.6 (BP = 1.000 ratio = 1.100 hyp_len = 296846 ref_len = 269739)
dev     BLEU = 16.39 44.9/21.2/11.4/6.6 (BP = 1.000 ratio = 1.207 hyp_len = 29302 ref_len = 24281)
test    BLEU = 18.91 47.0/23.4/13.6/8.5 (BP = 1.000 ratio = 1.161 hyp_len = 30851 ref_len = 26563)
"""
