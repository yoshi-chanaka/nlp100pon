from nlp100pon_41 import extractParsedInfo, createAdjMx
from nlp100pon_42 import organizeSentenceInfo
from nlp100pon_48 import convertAdjMx2TransMx
import numpy as np
import itertools
import re


def nounMasking(organized_info):
    """
    ['これらを', '統合した', '知的システムを', '作る', '試みも', 'なされている',
    'ACT-Rでは', 'エキスパートの', '推論ルール を', '統計的学習を', '元に',
    'ニューラルネットワークや', '生成規則を通して', '生成する']

    ['[NOUN]を', '[NOUN]した', '[NOUN][NOUN]を', '作る', '[NOUN]も', 'なされている',
    '[NOUN][NOUN][NOUN]では', '[NOUN]の', '[NOUN][NOUN]を', '[NOUN][NOUN][NOUN]を','[NOUN]に',
    '[NOUN]や', '[NOUN][NOUN]を通して', '[NOUN]する']
    """
    phrases         = organized_info['surface']
    parts_of_speech = organized_info['pos']
    phrases_original = [''.join(phr_list) for phr_list in phrases]
    phrases_masked_all = []
    for phrase_list, pos_list in zip(phrases, parts_of_speech):
        phrase_masked = []
        for phr, pos in zip(phrase_list, pos_list):
            if pos == '名詞':
                phrase_masked.append('[NOUN]')
            else:
                phrase_masked.append(phr)
        phrases_masked_all.append(''.join(phrase_masked))

    return phrases_original, phrases_masked_all

# 3.5 包含/排他的範囲にはbeginとendを使う
def nounReplacedPhrase(
    phrases_original, phrases_masked,
    idx_begin, idx_end, depend_trans_mx,
    replace_chars=['X', 'Y'], tail=False
    ):
    # idx_beginからidx_endまでの係り受け情報を取得
    binary_trans_info   = depend_trans_mx[idx_begin][idx_begin: idx_end + 1]
    trans_indices       = np.where(binary_trans_info == 1)[0]

    # 元の形態素列と名詞を[NOUN]に変換した形態素列を取得
    original_morphs = phrases_original[idx_begin: idx_end + 1][trans_indices]
    masked_morphs   = phrases_masked[idx_begin: idx_end + 1][trans_indices]

    replaced_phrase = original_morphs.copy()
    replaced_phrase[0] = re.sub(r'\[NOUN.*\]', replace_chars[0], masked_morphs[0])
    if tail:
        replaced_phrase[-1] = re.sub(r'\[NOUN.*\]', replace_chars[-1], masked_morphs[-1])
    return replaced_phrase

if __name__ == "__main__":
    SENTENCES_INFO = extractParsedInfo()

    path = '../../materials/chap05_49.txt'
    f = open(path, 'w', encoding='utf-8')

    for info_in_sentence in SENTENCES_INFO:

        organized_info_in_sentence  = organizeSentenceInfo(info_in_sentence)
        num_chunks = len(organized_info_in_sentence['surface'])

        depend_adj_mx   = createAdjMx(organized_info_in_sentence['dependencies'], num_chunks)
        depend_trans_mx = convertAdjMx2TransMx(depend_adj_mx, self_loop=True)

        phrases_original, phrases_masked_all = nounMasking(organized_info_in_sentence)
        phrases_original, phrases_masked_all = np.array(phrases_original), np.array(phrases_masked_all)

        pos_list = organized_info_in_sentence['pos']
        phrase_idx_combinations = list(itertools.combinations(range(num_chunks), 2))
        # itertools.combination()への入力がソートされているので，src < dst
        # 11章 一度に一つのことを：名詞のインデックスペアのみをまずは抽出
        # 7.7 ネストを浅くする
        noun_phrase_idx_combinations = \
            [(idx1, idx2) for idx1, idx2 in phrase_idx_combinations
                if ('名詞' in pos_list[idx1]) and ('名詞' in pos_list[idx2])]
        
        # 4.2 一貫性のある簡潔な改行位置：似ているコードは似ているように見せる
        for idx1, idx2 in noun_phrase_idx_combinations:

            if depend_trans_mx[idx1][idx2] == 1:
                # 文節iから構文木の根に至る経路上に文節jが存在する場合: 文節iから文節jのパスを表示
                # nounReplacedPhrase()
                replaced_phrase_list = [
                    nounReplacedPhrase(
                        phrases_original, phrases_masked_all,
                        idx_begin=idx1, idx_end=idx2,
                        depend_trans_mx=depend_trans_mx,
                        replace_chars=['X', 'Y'], tail=True
                    )
                ]

            else:
                # 文節iと文節jから構文木の根に至る経路上で共通の文節kで交わる場合:
                # 文節iから文節kに至る直前のパスと文節jから文節kに至る直前までのパス，文節kの内容を” | “で連結して表示
                sum_trans_row = depend_trans_mx[idx1] + depend_trans_mx[idx2]
                idx_common = sum_trans_row.tolist().index(2)

                replaced_phrase_list = []

                replaced_phrase_list.append(
                    nounReplacedPhrase(
                        phrases_original, phrases_masked_all,
                        idx_begin=idx1, idx_end=idx_common,
                        depend_trans_mx=depend_trans_mx,
                        replace_chars=['X', None], tail=False
                    )[:-1]
                )
                replaced_phrase_list.append(
                    nounReplacedPhrase(
                        phrases_original, phrases_masked_all,
                        idx_begin=idx2, idx_end=idx_common,
                        depend_trans_mx=depend_trans_mx,
                        replace_chars=['Y', None], tail=False
                    )[:-1]
                )
                common_phrase = phrases_original[idx_common]
                replaced_phrase_list.append([common_phrase])

            phrases_list = [' -> '.join(phr) for phr in replaced_phrase_list]
            f.write(' | '.join(phrases_list) + '\n')
    f.close()

    path = '../../materials/chap05_49.txt'
    with open(path, 'r', encoding='utf-8') as f:
        for i in range(10):
            print(f.readline().strip())
"""
X | Yのう | 語
X | Y -> エーアイとは | 語
X | Yとは | 語
X | Y -> という -> 道具を -> 用いて -> 研究する -> 計算機科学 -> の -> 一分野を -> 指す | 語
X | Yと -> 道具を -> 用いて -> 研究する -> 計算機科学 -> の -> 一分野を -> 指す | 語
X | Y -> という -> 道具を -> 用いて -> 研究する -> 計算機科学 -> の -> 一分野を -> 指す | 語
X | Yを -> 用いて -> 研究する -> 計算機科学 -> の -> 一分野を -> 指す | 語
X | Yを -> 研究する -> 計算機科学 -> の -> 一分野を -> 指す | 語
X | Yする -> 計算機科学 -> の -> 一分野を -> 指す | 語
X | Y -> の -> 一分野を -> 指す | 語
"""