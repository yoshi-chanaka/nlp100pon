from nlp100pon_41 import extractParsedInfo, createAdjMx
from nlp100pon_42 import organizeSentenceInfo
import numpy as np



def convertAdjMx2TransMx(adj_mx, self_loop=False):

    # adj_mx は正方行列
    trans_mx = adj_mx.copy()
    dst_indices_list = [row.tolist().index(1) if row.sum() >= 1 else -1 for row in adj_mx]

    for src_idx, dst_idx in reversed(list(enumerate(dst_indices_list))):
        if dst_idx >= 0:
            trans_mx[src_idx] += trans_mx[dst_idx]

    if self_loop:
        trans_mx += np.eye(trans_mx.shape[0])

    return trans_mx

# 6.5 入出力のコーナーケースに実例を使う
"""
（例）
入力:
0番目は2番目の文節に係っている
2番目は3番目の文節に係っている
3番目は5番目の文節に係っている
[[0. 0. 1. 0. 0. 0.]
[0. 0. 1. 0. 0. 0.]
[0. 0. 0. 1. 0. 0.]
[0. 0. 0. 0. 0. 1.]
[0. 0. 0. 0. 0. 1.]
[0. 0. 0. 0. 0. 0.]]
出力:
0番目の行で0-> 2-> 3 -> 5のパスがとれる（単位行列たす）
[[0. 0. 1. 1. 0. 1.],
[0. 0. 1. 1. 0. 1.],
[0. 0. 0. 1. 0. 1.],
[0. 0. 0. 0. 0. 1.],
[0. 0. 0. 0. 0. 1.],
[0. 0. 0. 0. 0. 0.]]

"""

if __name__ == "__main__":
    SENTENCES_INFO = extractParsedInfo()

    path = '../../materials/chap05_48.txt'
    with open(path, 'w', encoding='utf-8') as f:

        for info_in_sentence in SENTENCES_INFO:

            organized_info_in_sentence  = organizeSentenceInfo(info_in_sentence)
            num_chunks = len(organized_info_in_sentence['surface'])

            depend_adj_mx   = createAdjMx(organized_info_in_sentence['dependencies'], num_chunks)
            depend_trans_mx = convertAdjMx2TransMx(depend_adj_mx, self_loop=True)

            phrases_in_sentence = np.array([''.join(morphs) for morphs in organized_info_in_sentence['surface']])
            pos_list = organized_info_in_sentence['pos']
            for binary_trans_info in depend_trans_mx:

                if binary_trans_info.sum() > 1 and '名詞' in pos_list[binary_trans_info.tolist().index(1)]:
                    # 長さ2以上(1回以上の遷移)があれば行の値の和は2以上になる
                    # sum([1. 0. 1. 1. 0. 1.]) =  4
                    trans_indices = np.where(binary_trans_info==1)[0]
                    # [1. 0. 1. 1. 0. 1.] -> [0, 2, 3, 5]
                    f.write(' -> '.join(phrases_in_sentence[trans_indices]) + '\n')

    path = '../../materials/chap05_48.txt'
    with open(path, 'r', encoding='utf-8') as f:
        for i in range(10):
            print(f.readline().strip())

"""
人工知能 -> 語 -> 研究分野とも -> される
じんこうちのう -> 語 -> 研究分野とも -> される
AI -> エーアイとは -> 語 -> 研究分野とも -> される
エーアイとは -> 語 -> 研究分野とも -> される
計算 -> という -> 道具を -> 用いて -> 研究する -> 計算機科学 -> の -> 一分野を -> 指す -> 語 -> 研究分野とも -> される
概念と -> 道具を -> 用いて -> 研究する -> 計算機科学 -> の -> 一分野を -> 指す -> 語 -> 研究分野とも -> される
コンピュータ -> という -> 道具を -> 用いて -> 研究する -> 計算機科学 -> の -> 一分野を -> 指す -> 語 -> 研究分野とも -> される
道具を -> 用いて -> 研究する -> 計算機科学 -> の -> 一分野を -> 指す -> 語 -> 研究分野とも -> される
知能を -> 研究する -> 計算機科学 -> の -> 一分野を -> 指す -> 語 -> 研究分野とも -> される
研究する -> 計算機科学 -> の -> 一分野を -> 指す -> 語 -> 研究分野とも -> される
"""