from nlp100pon_95_sp_tokenize import tokenize

"""
95問目で学習したsentencepieceモデルでJPCのデータをトークナイズ
"""

if __name__ == "__main__":
    """
    タブ区切りで以下のような内容が書かれている

    000-lhr.web.wox.cc
    000-lhr.web.wox.cc
    0.750
    Also, it will always be hidden when becoming a premium user .
    また、 プレミアムユーザー になると常に非表示になります。
    """
    in_filepath = '../data/jpc-data/en-ja/en-ja.bicleaner05.txt'
    th_score = 0.7
    kind_data = ['train', 'dev', 'test']
    max_num_dict = {'train': 300000, 'dev': 2000, 'test': 2000}

    data_idx = 0 # 0: train, 1: dev, 2: test
    cnt = 0
    max_num = max_num_dict[kind_data[data_idx]]

    out_filepath_en = f'../data/jpc-data/jpc-{kind_data[data_idx]}.en'
    out_filepath_ja = f'../data/jpc-data/jpc-{kind_data[data_idx]}.ja'

    f_en = open(out_filepath_en, 'w')
    f_ja = open(out_filepath_ja, 'w')


    with open(in_filepath, 'r', encoding='utf-8') as f:
        line = f.readline()
        while line:
            info = line.strip().split('\t')
            line = f.readline()
            score = float(info[2])

            if score >= th_score:
                f_en.write(info[3] + '\n')
                f_ja.write(info[4] + '\n')
                cnt += 1

            if cnt >= max_num:

                f_en.close()
                f_ja.close()

                data_idx += 1
                cnt = 0

                if data_idx <= 2:
                    max_num = max_num_dict[kind_data[data_idx]]
                    out_filepath_en = f'../data/jpc-data/jpc-{kind_data[data_idx]}.en'
                    out_filepath_ja = f'../data/jpc-data/jpc-{kind_data[data_idx]}.ja'

                    f_en = open(out_filepath_en, 'w')
                    f_ja = open(out_filepath_ja, 'w')
                else:
                    break


    dirpath = '../data/jpc-data/'
    for lang in ['ja', 'en']:

        modelpath = f'../models/kftt_sp_{lang}.model'
        # tokenize
        for kind in ['train', 'dev', 'test']:
            tokenize(
                input_filepath=dirpath + f'jpc-{kind}.{lang}',
                output_filepath=f'../data/{kind}_{lang}.jpc',
                modelpath=modelpath
            )
