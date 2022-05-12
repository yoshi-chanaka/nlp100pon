from nlp100pon_41 import extractParsedInfo
from nlp100pon_42 import organizeSentenceInfo
import numpy as np

# 4.7 コードを段落に分ける
# p.121 定義の位置を下げる
# extractVerbCasePatternsでは，入力organized_infoから値を取ってくることが多いが，
# すべてを関数の先頭で定義せず，その変数を使う直前で定義する
def extractVerbCasePatterns(organized_info):
    """
    入力:
    {
        'surface': [[人工, 知能, ...], [言語, の, ...], ...]
        'base': [...],
        'pos': [...],
        'pos1': [...],
        'dependencies': [[0, 1], [1, 2], ...]
    }
    """

    NUM_PHRASES = len(organized_info['surface']) # 1文の文節の数
    base_morphs = organized_info['base']

    # 文節の位置と最左の動詞のみ記録
    verb_case_patterns = {}
    for idx in range(NUM_PHRASES):

        verb_case_patterns[str(idx)] = {
                'verb'              : None, # 最左の動詞の基本形
                'source_post_posit'   : [],   # 助詞
                'source_phrase'     : []    # 文節
            }

        pos_list = organized_info['pos'][idx]
        if '動詞' in pos_list:
            verb_index = pos_list.index('動詞')
            verb_case_patterns[str(idx)]['verb'] = base_morphs[idx][verb_index]


    # 係り受け関係をもとに文節の係り元を特定し，係元の助詞があれば記録
    surface_phrases = organized_info['surface']
    dependencies    = organized_info['dependencies']

    for src_idx, dst_idx in dependencies:

        src_pos_list = np.array(organized_info['pos'][src_idx])
        post_posit_indices = np.where(src_pos_list == '助詞')[0].tolist() # 係元文節の助詞のindex

        for pp_idx in post_posit_indices:
            verb_case_patterns[str(dst_idx)]['source_post_posit'].append(surface_phrases[src_idx][pp_idx])
            verb_case_patterns[str(dst_idx)]['source_phrase'].append(''.join(surface_phrases[src_idx]))

    return verb_case_patterns


if __name__ == "__main__":

    SENTENCES_INFO = extractParsedInfo()

    path = '../../materials/chap05_45.txt'
    with open(path, 'w', encoding='utf-8') as f:

        for info_in_sentence in SENTENCES_INFO:
            organized_info_in_sentence  = organizeSentenceInfo(info_in_sentence)
            verb_case_patterns          = extractVerbCasePatterns(organized_info_in_sentence)

            for _, pattern_dict in verb_case_patterns.items():
                source_post_posit = list(set(pattern_dict['source_post_posit']))
                source_post_posit.sort()
                if pattern_dict['verb'] == None or len(source_post_posit) == 0:
                    pass
                else:
                    f.write('{}\t{}\n'.format(pattern_dict['verb'], ' '.join(source_post_posit)))

    path = '../../materials/chap05_45.txt'
    with open(path, 'r', encoding='utf-8') as f:
        for i in range(10):
            print(f.readline().strip())
"""
用いる  を
する    て を
指す    を
代わる  に を
行う    て に
する    と も
述べる  で に の は
する    で を
する    を
する    を
"""


# sort ../../materials/chap05_45.txt | uniq -c | sort -nr > ../../materials/chap05_45_sorted.txt
"""
     49 する	を
     19 する	が
     15 する	に
     15 する	と
     12 する	は を
     10 する	に を
      9 よる	に
      9 する	で を
      8 行う	を
      8 する	が に
      6 呼ぶ	と
      6 基づく	に
      6 する	と は
    ...
"""
# grep -E "^行う" ../../materials/chap05_45.txt >> ../../materials/chap05_45_okonau.txt
# sort ../../materials/chap05_45_okonau.txt | uniq -c | sort -nr > ../../materials/chap05_45_okonau_sorted.txt
"""
      8 行う	を
      1 行う	まで を
      1 行う	は を をめぐって
      1 行う	は を
      1 行う	に を
      1 行う	に まで を
      1 行う	に により を
      1 行う	に
      1 行う	で を
      1 行う	で に を
      ...
"""
# grep -E "^なる" ../../materials/chap05_45.txt >> ../../materials/chap05_45_naru.txt
# sort ../../materials/chap05_45_naru.txt | uniq -c | sort -nr > ../../materials/chap05_45_naru_sorted.txt
"""
      6 なる	に は
      6 なる	が と
      4 なる	に
      4 なる	と
      2 なる	も
      2 なる	は
      2 なる	に は も
      2 なる	で は
      2 なる	で に は
      2 なる	で と など は
      ...
"""
# grep -E "^与える" ../../materials/chap05_45.txt >> ../../materials/chap05_45_ataeru.txt
# sort ../../materials/chap05_45_ataeru.txt | uniq -c | sort -nr > ../../materials/chap05_45_ataeru_sorted.txt
"""
      1 与える	に は を
      1 与える	が に
      1 与える	が など に
"""