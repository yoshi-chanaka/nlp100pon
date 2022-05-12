import re
import numpy as np
from nlp100pon_40 import Morph


class Chunk():

    def __init__(self, morphs=None, dst=None, srcs=None):

        self.morphs = morphs    # 形態素（Morphオブジェクト）のリスト
        self.dst = dst          # 係り先文節インデックス番号
        self.srcs = srcs        # 係り元文節インデックス番号のリスト

# 2.6 名前のフォーマットで情報を伝える
# 関数名：先頭が小文字で続く単語の頭文字は大文字
# クラス名：先頭のみ大文字
# 4.7 コードを段落に分ける

def extractParsedInfo(path = '../../data/ai.ja.txt.parsed'):

    # 11章：1度に1つのことを適用
    # 一度処理しやすい形に：「一文の解析結果のリスト」のリスト
    info_all_sentence_list = []
    with open(path, 'r', encoding='utf-8') as f:
        info_in_sentence = []
        line = f.readline()
        while line:
            line = line.strip()
            if line == 'EOS':
                info_all_sentence_list.append(info_in_sentence)
                info_in_sentence = []
            else:
                info_in_sentence.append(line)
            line = f.readline()
    """
    1文目
    ['* 0 17D 1/1 0.388993',
    '人工\t名詞,一般,*,*,*,*,人工,ジンコウ,ジンコー',
    '知能\t名詞,一般,*,*,*,*,知能,チノウ,チノー',
    '* 1 17D 2/3 0.613549',
    '（\t記号,括弧開,*,*,*,*,（, （,（']
    """

    # （係元, 係先）情報と形態素解析の情報を分離
    # Morph使う
    separated_info_all_sentence = []
    for info_in_sentence in info_all_sentence_list:
        separated_info_dict = {
            'destinations': [],
            'morphemes': []
        }
        for line in info_in_sentence:
            if len(re.findall(r'^\*', line)):
                # * 4 5D 2/2 1.035972 => 5
                splitted = line.split()
                src, dst = int(splitted[1]), int(splitted[2][:-1])
                separated_info_dict['destinations'].append([src, dst])
                separated_info_dict['morphemes'].append([])
            else:
                # 人工	名詞,一般,*,*,*,*,人工,ジンコウ,ジンコー
                separated_info_dict['morphemes'][-1].append(Morph(line))
        separated_info_all_sentence.append(separated_info_dict)
    """
    1文目
    [(0, 17), (1, 17), (2, 3), (3, 17), (4, 5), ...]
    [[<nlp100pon_40.Morph object at 0x7f0a6af44820>,
    <nlp100pon_40.Morph object at 0x7f0a6af44880>],
    [<nlp100pon_40.Morph object at 0x7f0a6af448e0>,
    <nlp100pon_40.Morph object at 0x7f0a6af44910>,
    <nlp100pon_40.Morph object at 0x7f0a6af44970>,
    <nlp100pon_40.Morph object at 0x7f0a6af449d0>, ...
    """
    SENTENCES_ALL_INFO = []
    for sentence_info_dict in separated_info_all_sentence:

        SENTENCES_ALL_INFO.append([])
        num_nodes = len(sentence_info_dict['morphemes'])
        adj_mx = createAdjMx(sentence_info_dict['destinations'], num_nodes)

        for idx, morphs in enumerate(sentence_info_dict['morphemes']):
            dst     = sentence_info_dict['destinations'][idx][1]
            srcs    = np.where(adj_mx[:, idx]==1)[0].tolist()
            chunk   = Chunk(morphs=morphs, dst=dst, srcs=srcs)
            SENTENCES_ALL_INFO[-1].append(chunk)

    return SENTENCES_ALL_INFO

def createAdjMx(nodes_list, n):
    adj = np.zeros((n, n))
    for idx_src, idx_dst in nodes_list:
        if idx_dst >= 0:
            adj[idx_src][idx_dst] = 1
    return adj


if __name__ == "__main__":
    sentences_info = extractParsedInfo()
    print(sentences_info[1])
    for info in sentences_info[1][:5]:
        print(info.__dict__)
        print('\t', info.morphs[0].__dict__)
"""
    * 0 17D 1/1 0.388993
    人工	名詞,一般,*,*,*,*,人工,ジンコウ,ジンコー
    知能	名詞,一般,*,*,*,*,知能,チノウ,チノー
    * 1 17D 2/3 0.613549
    （	記号,括弧開,*,*,*,*,（,（,（
    じん	名詞,一般,*,*,*,*,じん,ジン,ジン
    こうち	名詞,一般,*,*,*,*,こうち,コウチ,コーチ
    のう	助詞,終助詞,*,*,*,*,のう,ノウ,ノー
    、	記号,読点,*,*,*,*,、,、,、
    、	記号,読点,*,*,*,*,、,、,、
    * 2 3D 0/0 0.758984
    AI	名詞,一般,*,*,*,*,*
    * 3 17D 1/5 0.517898
    〈	記号,括弧開,*,*,*,*,〈,〈,〈
    エーアイ	名詞,固有名詞,一般,*,*,*,*
    〉	記号,括弧閉,*,*,*,*,〉,〉,〉
    ...
"""
"""
出力：
{'morphs': [<nlp100pon_40.Morph object at 0x7f0330946850>,
            <nlp100pon_40.Morph object at 0x7f03309468b0>],
'dst': 17, 'srcs': []}
    {'surface': '人工', 'base': '人工', 'pos': '名詞', 'pos1': '一般'}
{'morphs': [<nlp100pon_40.Morph object at 0x7f0330946910>,
            <nlp100pon_40.Morph object at 0x7f0330946940>,
            <nlp100pon_40.Morph object at 0x7f03309469a0>,
            <nlp100pon_40.Morph object at 0x7f0330946a00>,
            <nlp100pon_40.Morph object at 0x7f0330946a60>,
            <nlp100pon_40.Morph object at 0x7f0330946ac0>],
'dst': 17, 'srcs': []}
    {'surface': '（', 'base': '（', 'pos': '記号', 'pos1': '括弧開'}
{'morphs': [<nlp100pon_40.Morph object at 0x7f0330946b20>],
'dst': 3, 'srcs': []}
    {'surface': 'AI', 'base': '*', 'pos': '名詞', 'pos1': '一般'}
{'morphs': [<nlp100pon_40.Morph object at 0x7f0330946b80>,
            <nlp100pon_40.Morph object at 0x7f0330946be0>,
            <nlp100pon_40.Morph object at 0x7f0330946c40>,
            <nlp100pon_40.Morph object at 0x7f0330946ca0>,
            <nlp100pon_40.Morph object at 0x7f0330946d00>,
            <nlp100pon_40.Morph object at 0x7f0330946d60>,
            <nlp100pon_40.Morph object at 0x7f0330946dc0>],
'dst': 17, 'srcs': [2]}
    {'surface': '〈', 'base': '〈', 'pos': '記号', 'pos1': '括弧開'}
{'morphs': [<nlp100pon_40.Morph object at 0x7f0330946e20>,
            <nlp100pon_40.Morph object at 0x7f0330946e80>,
            <nlp100pon_40.Morph object at 0x7f0330946ee0>],
'dst': 5, 'srcs': []}
    {'surface': '「', 'base': '「', 'pos': '記号', 'pos1': '括弧開'}
"""
