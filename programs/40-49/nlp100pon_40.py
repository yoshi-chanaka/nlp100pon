import re

class Morph():

    def __init__(self, line):

        # 4.4 縦の線をまっすぐにする
        surface, info = line.split('\t', 1)
        splitted_info = info.split(',')

        self.surface    = surface              # 表層形
        self.base       = splitted_info[6]     # 基本形
        self.pos        = splitted_info[0]     # 品詞
        self.pos1       = splitted_info[1]     # 品詞細分類1

if __name__ == "__main__":
    """
    * 0 -1D 1/1 0.000000
    人工	名詞,一般,*,*,*,*,人工,ジンコウ,ジンコー
    知能	名詞,一般,*,*,*,*,知能,チノウ,チノー
    EOS
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
    ...
    """

    sentences, morphs = [], []

    path = '../../data/ai.ja.txt.parsed'
    f = open(path, 'r', encoding='utf-8')
    line = f.readline()

    while line:
        line = line.strip()

        if len(re.findall(r'^\*\s[0-9]+\s', line)):
            # * 0 -1D 1/1 0.000000
            pass
        elif line == 'EOS':
            sentences.append(morphs)
            morphs = []
        else:
            # 人工	名詞,一般,*,*,*,*,人工,ジンコウ,ジンコー
            morphs.append(Morph(line))

        line = f.readline()

    f.close()


    for morph in sentences[1][:10]:
        print(morph.__dict__)

"""
{'surface': '人工', 'base': '人工', 'pos': '名詞', 'pos1': '一般'}
{'surface': '知能', 'base': '知能', 'pos': '名詞', 'pos1': '一般'}
{'surface': '（', 'base': '（', 'pos': '記号', 'pos1': '括弧開'}
{'surface': 'じん', 'base': 'じん', 'pos': '名詞', 'pos1': '一般'}
{'surface': 'こうち', 'base': 'こうち', 'pos': '名詞', 'pos1': '一般'}
{'surface': 'のう', 'base': 'のう', 'pos': '助詞', 'pos1': '終助詞'}
{'surface': '、', 'base': '、', 'pos': '記号', 'pos1': '読点'}
{'surface': '、', 'base': '、', 'pos': '記号', 'pos1': '読点'}
{'surface': 'AI', 'base': '*', 'pos': '名詞', 'pos1': '一般'}
{'surface': '〈', 'base': '〈', 'pos': '記号', 'pos1': '括弧開'}
"""
