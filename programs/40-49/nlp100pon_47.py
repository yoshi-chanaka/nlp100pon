from nlp100pon_41 import extractParsedInfo
from nlp100pon_42 import organizeSentenceInfo
import numpy as np

# 4.7 コードを段落に分ける
# 6.6 コードの意図を書く
def extractSahenVerbSyntax(organized_info):
    """
    入力：
    {
        'surface': [[人工, 知能, ...], [言語, の, ...], ...]
        'base': [...],
        'pos': [...],
        'pos1': [...],
        'dependencies': [[0, 1], [1, 2], ...]
    }
    """

    NUM_PHRASES = len(organized_info['surface']) # 1文の文節の数
    verb_case_patterns = {}

    for idx in range(NUM_PHRASES):
        verb_case_patterns[str(idx)] = {
                'verb'              : None, # 最左の動詞の基本形
                'source_post_posit' : [],   # 助動詞
                'source_phrase'     : [],   # 文節
                'sahen'             : None  # サ変接続名詞の位置
        }

    for src_idx, dst_idx in organized_info['dependencies']:

        # 「サ変接続名詞 + を」を見つける: srcを参照
        src_base_list   = organized_info['base'][src_idx]   + ['dummy']
        src_pos_list    = organized_info['pos'][src_idx]    + ['dummy']
        src_pos1_list   = organized_info['pos1'][src_idx]
        sahen_idx = -1

        while 'サ変接続' in src_pos1_list:
            sahen_idx = src_pos1_list.index('サ変接続')

            if src_base_list[sahen_idx + 1] == 'を' and src_pos_list[sahen_idx + 1] == '助詞':
                break
            else:
                src_pos1_list[sahen_idx] = 'replaced'
                sahen_idx = -1

        # 上記の文節のうち，係先が動詞となっている場合のみパターンとして記録: dstを参照
        dst_base_list   = organized_info['base'][dst_idx]
        dst_pos_list    = organized_info['pos'][dst_idx]

        if '動詞' in dst_pos_list and sahen_idx >= 0:
            verb_index = dst_pos_list.index('動詞')
            sahen_syntax = src_base_list[sahen_idx] + src_base_list[sahen_idx+1] + dst_base_list[verb_index]

            verb_case_patterns[str(dst_idx)]['verb'] = sahen_syntax
            verb_case_patterns[str(dst_idx)]['sahen'] = src_idx


    # 係り受け関係をもとに文節の係り元を特定し，係元の助詞があれば記録
    for src_idx, dst_idx in organized_info['dependencies']:

        phrases = organized_info['surface']
        src_pos_list = np.array(organized_info['pos'][src_idx])
        post_posit_indices = np.where(src_pos_list == '助詞')[0].tolist() # 助詞のindex

        for pp_idx in post_posit_indices:
            if src_idx != verb_case_patterns[str(dst_idx)]['sahen']:
                verb_case_patterns[str(dst_idx)]['source_post_posit'].append(phrases[src_idx][pp_idx])
                verb_case_patterns[str(dst_idx)]['source_phrase'].append(''.join(phrases[src_idx]))

    return verb_case_patterns

if __name__ == "__main__":

    SENTENCES_INFO = extractParsedInfo()
    path = '../../materials/chap05_47.txt'
    with open(path, 'w', encoding='utf-8') as f:

        for info_in_sentence in SENTENCES_INFO:
            organized_info_in_sentence  = organizeSentenceInfo(info_in_sentence)
            verb_case_patterns          = extractSahenVerbSyntax(organized_info_in_sentence)

            for _, pattern_dict in verb_case_patterns.items():
                source_post_posit   = pattern_dict['source_post_posit']
                source_phrase       = pattern_dict['source_phrase']

                if pattern_dict['verb'] == None or len(source_post_posit) == 0:
                    pass
                else:
                    f.write('{}\t{}\t{}\n'.format(
                        pattern_dict['verb'],
                        ' '.join(source_post_posit),
                        ' '.join(source_phrase)
                    ))

    path = '../../materials/chap05_47.txt'
    with open(path, 'r', encoding='utf-8') as f:
        for i in range(10):
            print(f.readline().strip())
"""
行動を代わる    に      人間に
記述をする      と      主体と
注目を集める    が      サポートベクターマシンが
学習を行う      を に   経験を 元に
学習をする      て で は を に を通して なされている ACT-Rでは ACT-Rでは 推論ルールを 元に 生成規則を通して
進化を見せる    て は て において       活躍している 敵対的生成ネットワークは 加えて 生成技術において
開発を行う      は      エイダ・ラブレスは
処理を行う      に により に    同年に ティム・バーナーズリーにより Webに
意味をする      に      データに
処理を行う      て に   付加して コンピュータに
"""