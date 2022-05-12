from nlp100pon_41 import extractParsedInfo
from nlp100pon_42 import organizeSentenceInfo
from nlp100pon_45 import extractVerbCasePatterns

# 問45で作ったextractVerbCasePatternsを使う
# 4.4 縦の線をまっすぐにする
if __name__ == "__main__":

    SENTENCES_INFO = extractParsedInfo()
    path = '../../materials/chap05_46.txt'
    with open(path, 'w', encoding='utf-8') as f:

        for info_in_sentence in SENTENCES_INFO:
            organized_info_in_sentence  = organizeSentenceInfo(info_in_sentence)
            verb_case_patterns          = extractVerbCasePatterns(organized_info_in_sentence)
            """
            verb_case_patternsへの入力:
            {
                'surface': [[人工, 知能, ...], [言語, の, ...], ...]
                'base': [...],
                'pos': [...],
                'pos1': [...],
                'dependencies': [[0, 1], [1, 2], ...]
            }
            """
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

    path = '../../materials/chap05_46.txt'
    with open(path, 'r', encoding='utf-8') as f:
        for i in range(10):
            print(f.readline().strip())
"""
用いる	を	道具を
する    て を   用いて 知能を
指す	を	一分野を
代わる  を に   知的行動を 人間に
行う	て に	代わって コンピューターに
する    と も   研究分野とも 研究分野とも
述べる  で は の に     解説で 佐藤理史は 次のように 次のように
する    を で   知的能力を コンピュータ上で
する	を	推論判断を
する	を	画像データを
"""