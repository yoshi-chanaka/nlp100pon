from mask import generate_square_subsequent_mask
from convert_sequence import convert_sent2idx
from process import post_process
from beam_search import BeamSearchNode

import torch
from heapq import heapify, heappop, heappush

def beam_decode_history(
    model,
    src,
    src_mask,
    max_len,
    device,
    bos_idx,
    eos_idx,
    beam_width=2,
    margin=30
):

    best_list, best_nll = [], []
    history = []

    src = src.to(device)
    src_mask = src_mask.to(device)
    memory = model.encode(src, src_mask).to(device)
    seq = torch.ones(1, 1).fill_(bos_idx).type(torch.long).to(device)

    # 初期化
    node = BeamSearchNode(token_ids=seq, logp=0, length=1)
    nodes = []
    heappush(nodes, (node.negative_log_likelihood(), id(node), node))

    for iter in range(max_len + margin):

        if len(best_list) >= beam_width:
            break

        # nllが小さいものからビームサイズ分取り出してnodesをリセット
        n_best_nodes = [heappop(nodes) for _ in range(min(len(nodes), beam_width))]
        nodes = []
        heapify(nodes)

        for _, _, node in n_best_nodes:

            # if node.token_ids.flatten()[-1] == eos_idx:
            #      continue

            # 次のトークンの予測
            seq = node.token_ids
            tgt_mask = (generate_square_subsequent_mask(seq.size(0)).type(torch.bool)).to(device)
            out = model.decode(seq, memory, tgt_mask).transpose(0, 1)
            prob = torch.softmax(model.generator(out[:, -1]), dim=1)
            topk_prob, topk_idx = torch.topk(prob, beam_width)

            # top-Kに対してheappush．eosが出たらpushしない
            for prob, next_tok in zip(topk_prob.squeeze(0), topk_idx.squeeze(0)):

                new_seq = torch.cat(
                    [seq, torch.ones(1, 1).type_as(src.data).fill_(next_tok)], dim=0)
                new_node = BeamSearchNode(
                    token_ids=new_seq,
                    logp=node.logp + torch.log(prob).item(),
                    length=node.length + 1
                )

                if next_tok == eos_idx:
                    best_list.append(new_seq.flatten().cpu())
                    best_nll.append(new_node.negative_log_likelihood())
                else:
                    heappush(nodes, (new_node.negative_log_likelihood(), id(new_node), new_node))
                history.append(new_node.token_ids.flatten().cpu())

    for i in range(beam_width - len(best_list)):
        _, _, node = heappop(nodes)
        best_list.append(node.token_ids.flatten().cpu())
        best_nll.append(node.negative_log_likelihood())

    return best_list, best_nll, history


if __name__ == "__main__":
    from load import load_data
    from translate import load_model
    import sentencepiece as spm

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using {} device'.format(device))

    _, vocab_src, vocab_tgt = load_data(only_vocab=True)
    model = load_model(device=device)
    sp_tokenizer = spm.SentencePieceProcessor(model_file='../models/kftt_sp_ja.model')

    beam_width = 3
    input_text = "今日は良い天気ですね。"
    # input_text = "私の考える良い研究テーマの決め方について書きます。"
    src_sentence = ' '.join(sp_tokenizer.EncodeAsPieces(input_text))
    print('\n' + src_sentence)

    src = torch.Tensor(convert_sent2idx(src_sentence.split(
        ' '), vocab_src, set(list(vocab_src.keys())))).long()
    src = src.unsqueeze(0).reshape(-1, 1)
    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)

    model.eval()
    best_list, nll_list, history = beam_decode_history(
        model, src, src_mask,
        max_len=num_tokens, device=device, bos_idx=2, eos_idx=3,
        beam_width=beam_width
    )
    vocab_decode = list(vocab_tgt.keys())
    for tgt_tokens, score in zip(best_list, nll_list):
        output = " ".join([vocab_decode[idx.item()] for idx in tgt_tokens])
        print(score, post_process(output))
    print(len(best_list))



    from graphviz import Digraph

    dg = Digraph(format='png')
    edges = []
    for tgt_tokens in history:
        prev_n = None
        for depth, tok in enumerate(tgt_tokens):
            n = str(depth) + '-' + str(tok.item())
            dg.node(n, label=vocab_decode[tok.item()])
            if prev_n != None and ((prev_n, n) not in edges):
                dg.edge(prev_n, n)
                edges.append((prev_n, n))
            prev_n = n

    dg.render('../figures/beam_tree')

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
3
"""
