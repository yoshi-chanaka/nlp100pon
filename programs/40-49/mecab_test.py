import MeCab

tagger = MeCab.Tagger()
sentence = input()
if sentence == '':
    sentence = '太郎は花子が読んでいる本を次郎に渡した。'
res = tagger.parse(sentence)
print(res)
