from nlp100pon_41 import extractParsedInfo


def organizeSentenceInfo(info_in_sentence):

    # 品詞は問43で，品詞細分類は問47で使う
    organized_info_in_sentence = {
        'surface': [],
        'base': [],
        'pos': [],
        'pos1': []
    }
    dependence_edges = []

    for idx_src, chunk_info in enumerate(info_in_sentence):
        """
        chunk_info :
            {'morphs': [<nlp100pon_40.Morph object at 0x7f0330946850>,
                        <nlp100pon_40.Morph object at 0x7f03309468b0>],
            'dst': 17, 'srcs': []}
        """
        chunk_info_dict = {
            'surface': [],
            'base': [],
            'pos': [],
            'pos1': []
        }

        for morph in chunk_info.morphs:
            if morph.pos != '記号':
                chunk_info_dict['surface'].append(morph.surface)
                chunk_info_dict['base'].append(morph.base)
                chunk_info_dict['pos'].append(morph.pos)
                chunk_info_dict['pos1'].append(morph.pos1)

        organized_info_in_sentence['surface'].append(chunk_info_dict['surface'])
        organized_info_in_sentence['base'].append(chunk_info_dict['base'])
        organized_info_in_sentence['pos'].append(chunk_info_dict['pos'])
        organized_info_in_sentence['pos1'].append(chunk_info_dict['pos1'])

        if chunk_info.dst >= 0:
            dependence_edges.append([idx_src, chunk_info.dst])
    organized_info_in_sentence['dependencies'] = dependence_edges

    return organized_info_in_sentence

if __name__ == "__main__":
    # 2.6 名前のフォーマットで情報を伝える
    # 大元の情報を持つ変数はすべて大文字で表記
    SENTENCES_INFO = extractParsedInfo()

    connected_phrase_pairs = []
    for info_in_sentence in SENTENCES_INFO:
        organized_info_in_sentence = organizeSentenceInfo(info_in_sentence)
        phrases = organized_info_in_sentence['surface']
        for src_idx, dst_idx in organized_info_in_sentence['dependencies']:
            src_phrase, dst_phrase = ''.join(phrases[src_idx]), ''.join(phrases[dst_idx])
            line = '{}\t{}'.format(src_phrase, dst_phrase)
            connected_phrase_pairs.append(line)

    print(len(connected_phrase_pairs))
    for line in connected_phrase_pairs[:20]:
        print(line)

"""
2861
人工知能        語
じんこうちのう  語
AI      エーアイとは
エーアイとは    語
計算    という
という  道具を
概念と  道具を
コンピュータ    という
という  道具を
道具を  用いて
用いて  研究する
知能を  研究する
研究する        計算機科学
計算機科学      の
の      一分野を
一分野を        指す
指す    語
語      研究分野とも
言語の  推論
理解や  推論
"""