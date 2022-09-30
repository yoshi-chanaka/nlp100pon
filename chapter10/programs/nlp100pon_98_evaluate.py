if __name__ == "__main__":
    """
    ファインチューニングしたモデルを使って翻訳
    """
    from load import load_data
    from translate import load_model, translate

    import sentencepiece as spm
    import torch
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using {} device'.format(device))

    _, vocab_ja, vocab_en = load_data(only_vocab=True)
    model = load_model(path = '../models/model_finetuned.pth', device=device)

    sp_tokenizer = spm.SentencePieceProcessor(model_file='../models/kftt_sp_ja.model')

    input_text = "今日は良い天気ですね。"
    proc_text = ' '.join(sp_tokenizer.EncodeAsPieces(input_text))
    print('\n' + proc_text)
    print(translate(model=model, src_sentence=proc_text, vocab_src=vocab_ja, vocab_tgt=vocab_en, device=device, post_proc=True))
    # Today, it is good weather.

    input_text = "私の考える良い研究テーマの決め方について書きます。"
    proc_text = ' '.join(sp_tokenizer.EncodeAsPieces(input_text))
    print('\n' + proc_text)
    print(translate(model=model, src_sentence=proc_text, vocab_src=vocab_ja, vocab_tgt=vocab_en, device=device, post_proc=True))
    # I will write on how the team should think of the team's own research team.

    input_text = "明日から再び研究室の活動が始まります。"
    proc_text = ' '.join(sp_tokenizer.EncodeAsPieces(input_text))
    print('\n' + proc_text)
    print(translate(model=model, src_sentence=proc_text, vocab_src=vocab_ja, vocab_tgt=vocab_en, device=device, post_proc=True))
    # The activities of the research room began to be started again from the Ming period.

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

