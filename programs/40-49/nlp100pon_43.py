from nlp100pon_41 import extractParsedInfo
from nlp100pon_42 import organizeSentenceInfo


if __name__ == "__main__":
    # 2.6 名前のフォーマットで情報を伝える
    # 大元の情報を持つ変数はすべて大文字で表記
    SENTENCES_INFO = extractParsedInfo()

    connected_noun_verb_pairs = []
    for info_in_sentence in SENTENCES_INFO:
        organized_info_in_sentence = organizeSentenceInfo(info_in_sentence)
        phrases, parts_of_speech = \
            organized_info_in_sentence['surface'], organized_info_in_sentence['pos']
        for src_idx, dst_idx in organized_info_in_sentence['dependencies']:
            if ('名詞' in parts_of_speech[src_idx]) and ('動詞' in parts_of_speech[dst_idx]):
                src_phrase, dst_phrase = ''.join(phrases[src_idx]), ''.join(phrases[dst_idx])
                line = '{}\t{}'.format(src_phrase, dst_phrase)
                connected_noun_verb_pairs.append(line)

    print(len(connected_noun_verb_pairs))
    for line in connected_noun_verb_pairs[:20]:
        print(line)

"""
1325
道具を  用いて
知能を  研究する
一分野を        指す
知的行動を      代わって
人間に  代わって
コンピューターに        行わせる
研究分野とも    される
解説で  述べている
佐藤理史は      述べている
次のように      述べている
知的能力を      実現する
コンピュータ上で        実現する
技術ソフトウェアコンピュータシステム    ある
応用例は        ある
推論判断を      模倣する
画像データを    解析して
解析して        検出抽出したりする
パターンを      検出抽出したりする
画像認識等が    ある
1956年に        命名された
"""