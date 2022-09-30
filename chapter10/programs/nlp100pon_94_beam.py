from convert_sequence import convert_sent2idx
from translate import translate
from beam_search import beam_decode
from process import post_process

import torch

if __name__ == "__main__":
    from load import load_data
    from translate import load_model
    import sentencepiece as spm

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using {} device'.format(device))

    _, vocab_ja, vocab_en = load_data(only_vocab=True)
    model = load_model(device=device)
    sp_tokenizer = spm.SentencePieceProcessor(model_file='../models/kftt_sp_ja.model')

    input_text_list = [
        "今日は良い天気ですね。",
        "私の考える良い研究テーマの決め方について書きます。",
        "明日から再び研究室の活動が始まります。"
    ]

    for input_text in input_text_list:
        src_sentence = ' '.join(sp_tokenizer.EncodeAsPieces(input_text))
        print('\n' + src_sentence)
        # src_sentence = "▁また 、 通 親 の子 、 源 通 宗 または 久我 通 光 を 父 親 とする説もある 。"

        beam_width = 3
        src = torch.Tensor(convert_sent2idx(src_sentence.split(
            ' '), vocab_ja, set(list(vocab_ja.keys())))).long()
        src = src.unsqueeze(0).reshape(-1, 1)
        num_tokens = src.shape[0]
        src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)

        model.eval()
        best_list, nll_list = beam_decode(
            model, src, src_mask,
            max_len=num_tokens + 30, device=device, bos_idx=2, eos_idx=3,
            beam_width=beam_width, num_return=beam_width
        )
        vocab_decode = list(vocab_en.keys())
        for tgt_tokens, score in zip(best_list, nll_list):
            output = " ".join([vocab_decode[idx.item()] for idx in tgt_tokens])
            print(score, post_process(output))

        print(translate(
                model=model,
                src_sentence=src_sentence,
                vocab_src=vocab_ja,
                vocab_tgt=vocab_en,
                device=device,
                bos_idx=2,
                eos_idx=3,
                post_proc=True,
                beam_width=3,
                method='beam'
        ))

"""
Using cuda device
Loading Dataset ...
Done
Loading Model ... : Using cuda device
Done

▁ 今日 は 良い 天 気 で す ね 。
0.7097573520188414 <bos> It is good weather today.<eos>
0.5695042295482696 <bos> Today, it is good weather.<eos>
0.715904607960466 <bos> Today, it is good at weather.<eos>
Today, it is good weather.

▁ 私 の 考え る 良い 研究 テ ー マ の 決め 方 について 書き ます 。
0.8148587296646116 <bos> I will write on how to decide themes of good research laboratory.<eos>
0.8583983777004559 <bos> I will write on how to decide themes of good research laborers.<eos>
0.8670948364934371 <bos> I will write on how to decide themes of good research laborator.<eos>
I will write on how to decide themes of good research laboratory.

▁ 明 日 から 再び 研究 室 の 活動 が 始まり ます 。
0.5154895344622584 <bos> The activities of research laboratory started again from the Ming period.<eos>
0.7485555189852374 <bos> The activities of research laboratory started again from the Ming period began.<eos>
0.6035182998607872 <bos> The activities of research laboratory began to be started from the Ming period.<eos>
0.7773570289086198 <bos> The activities of research laboratory began to be started from the Ming Dynasty.<eos>
The activities of research laboratory started again from the Ming period.
"""
