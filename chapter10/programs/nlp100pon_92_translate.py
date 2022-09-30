if __name__ == "__main__":
    
    from load import load_data
    from translate import load_model, translate

    import sentencepiece as spm
    import torch
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using {} device'.format(device))

    _, vocab_ja, vocab_en = load_data(only_vocab=True)
    model = load_model(path = '../models/model.pth', device=device)

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
