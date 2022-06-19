import numpy as np

# 6.6 コードの意図を書く：入力テキストに対して複数の処理をしているのでコメントをつける
def PreprocessingText(input_text, stop_words):

    import re

    processed_text = input_text.lower()

    # 記号の削除
    pattern = r'[,\.\:\;\!\?\-“”@&\(\)\[\]\{\}\'\"\t\/…\*—\+\|_]'
    processed_text = re.sub(pattern, ' ', processed_text)

    # 数字の分離
    for n in range(10):
        processed_text = f' {str(n)} '.join(processed_text.split(str(n)))

    # アルファベット1文字の削除
    processed_text = re.sub(r'\s[a-z]\s', ' ', processed_text)

    # 2つ以上のスペースの連続を1つのスペースに
    processed_text = re.sub(r'\s\s+', ' ', processed_text)

    # stop_wordsの削除
    if stop_words != None:
        remove_words    = set(stop_words)
        words_in_text   = processed_text.strip().split()
        processed_text  = ' '.join([w for w in words_in_text if w not in remove_words])

    return processed_text



def ComputeAvgEmbeddings(input_corpus, w2v_model):
    """
    入力例:
    [[angelina jolie angelina jolie tighten security brad pitt prank],
    [mila kunis mila kunis ashton kutcher wanted kids nearly year], ...]
    """

    sentence_embeddings = []
    for text in input_corpus:
        word_embeddings = []
        for word in text.split(' '):
            try:
                word_embeddings.append(w2v_model[word].tolist())
            except:
                pass
        sentence_embeddings.append(np.mean(word_embeddings, axis=0).tolist())

    return np.array(sentence_embeddings)



def CreateDataset(rm_stopwords=True):

    from nltk.corpus import stopwords
    import nltk
    nltk.download('stopwords')

    if rm_stopwords:
        stop_words = stopwords.words('english')
        for word in ['up', 'down', 'over', 'under']:
            stop_words.remove(word)

    category2label = {'b': 0, 't': 1, 'e': 2, 'm': 3}
    data_path = '../../data/NewsAggregatorDataset/' # train.txt, valid.txt, test.txt
    labels_dict, corpora_processed_dict = {}, {}

    for kind in ['train', 'valid', 'test']:

        labels_dict[kind] = []
        corpora_processed_dict[kind] = []

        f = open(data_path + kind + '.txt', 'r', encoding='utf-8')
        line = f.readline()
        while line:
            categ, text = line.strip().split('\t', 1)
            if categ in ['b', 't', 'e', 'm']:
                labels_dict[kind].append(category2label[categ])
                corpora_processed_dict[kind].append(PreprocessingText(text, stop_words))
            line = f.readline()
        f.close()

    features_dict = {}
    from gensim.models import KeyedVectors
    print('loading word2vec ... ')
    w2v_path = '../../data/GoogleNews-vectors-negative300.bin'
    w2v_model = KeyedVectors.load_word2vec_format(w2v_path, binary=True)
    print('done')
    for kind in ['train', 'valid', 'test']:
        sentence_embeddings = ComputeAvgEmbeddings(corpora_processed_dict[kind], w2v_model)
        features_dict[kind] = sentence_embeddings


    return features_dict, labels_dict

def LoadData(path):
    
    import pickle
    with open(path, 'rb') as f:
        data = pickle.load(f)
        
    X, y = data['features'], data['labels']

    return X, y



if __name__ == "__main__":

    X, Y = CreateDataset()
    save_data = {'features': X, 'labels': Y}
    save_path = '../../data/NewsAggregatorDataset/chap08_avgembed.pickle'

    import pickle
    with open(save_path, 'wb') as f:
        pickle.dump(save_data, f)
    
    X, Y = LoadData(save_path)
    for kind in ['train', 'valid', 'test']:
        print('{}\t{}, {}'.format(kind, X[kind].shape, len(Y[kind])))
    
    """
    train   (10672, 300), 10672
    valid   (1334, 300), 1334
    test    (1334, 300), 1334
    """