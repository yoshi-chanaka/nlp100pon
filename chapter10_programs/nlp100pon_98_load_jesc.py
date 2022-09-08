from nlp100pon_95_sp_tokenize import tokenize

import sentencepiece as spm

"""
95問目で学習したsentencepieceモデルでJESCのデータをトークナイズ
"""

if __name__ == "__main__":
    
    dirpath = '../data/jesc-data/'
    
    # ファイル内に英語文と日本語文がタブ区切りで入っているので別ファイルに分ける
    for filename in ['train', 'dev', 'test']:
        
        f_en = open(dirpath + f'jesc-{filename}.en', 'w')
        f_ja = open(dirpath + f'jesc-{filename}.ja', 'w')

        with open(dirpath + f'split/{filename}', 'r') as f:
            line = f.readline()
            while line:
                
                text_en, text_ja = line.strip().split('\t')
                f_en.write(text_en + '\n')
                f_ja.write(text_ja + '\n')
                line = f.readline()
        
        f_en.close()
        f_ja.close()
    

    for lang in ['ja', 'en']:

        modelpath = f'../models/kftt_sp_{lang}.model'
        # tokenize
        for kind in ['train', 'dev', 'test']:
            tokenize(
                input_filepath = dirpath + f'jesc-{kind}.{lang}',
                output_filepath = f'../data/{kind}_{lang}.jesc',
                modelpath = modelpath
            )

