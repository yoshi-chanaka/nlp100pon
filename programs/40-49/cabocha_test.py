import CaboCha

cp = CaboCha.Parser()
sentence = input()
if sentence == '':
    sentence = '太郎は花子が読んでいる本を次郎に渡した。'
res = cp.parseToString(sentence)
print(res)
