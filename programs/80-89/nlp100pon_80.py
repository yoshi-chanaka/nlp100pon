from chap09_labelencode import MakeLabelEncoder
import torch

def ConvertSentenceToIndices(sentence, word2id_encoder, vocab=None):
    """
    input
    sentence: ['今日', 'は', '良い', '天気', 'です', 'ね']
    """
    if vocab == None:
        indices = [word2id_encoder[w] for w in sentence]
    else:
        indices = [word2id_encoder[w] for w in sentence if w in vocab]

    return indices

def ConvertDatasetToIDSeqSet(corpora, word2id_encoder, torch_tensor=True):
    """
    input: corpora
    {
        'train': ['i have a pen', 'this is a pen', ...],
        'valid': [...],
        'test' : [...],
    }
    """
    id_seqsets_dict = {}
    for key in corpora.keys():
        corpus = corpora[key]
        if torch_tensor:
            id_seqsets_dict[key] = [torch.tensor(ConvertSentenceToIndices(sentence.split(), word2id_encoder)) for sentence in corpus]
        else:
            id_seqsets_dict[key] = [ConvertSentenceToIndices(sentence.split(), word2id_encoder) for sentence in corpus]

    return id_seqsets_dict

def PreprocessingText(input_text):

    import re

    processed_text = input_text.lower()

    # 記号の削除
    pattern = r'[,\.\:\-“”#@&\(\)\[\]\{\}\'\"\t\/…\*—\+\|_]' 
    processed_text = re.sub(pattern, ' ', processed_text)

    # 記号（!?）の分離
    for sym in ['!', '?']:
        processed_text = f' {sym} '.join(processed_text.split(sym))

    # 数字の分離
    for n in range(10):
        processed_text = f' {str(n)} '.join(processed_text.split(str(n)))

    # 2つ以上のスペースの連続を1つのスペースに
    processed_text = re.sub(r'\s\s+', ' ', processed_text)

    return processed_text.strip()


def LoadData():

    # 文章（単語列）, ラベル，labelencoderを返す

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
                corpora_processed_dict[kind].append(PreprocessingText(text))
            line = f.readline()
        f.close()

        labels_dict[kind] = torch.tensor(labels_dict[kind])

    corpus = corpora_processed_dict['train'] + corpora_processed_dict['valid'] + corpora_processed_dict['test']
    corpus = (' '.join(corpus)).split()
    label_encoder, _ = MakeLabelEncoder(corpus)

    return corpora_processed_dict, labels_dict, label_encoder

if __name__ == "__main__":
    
    corpora_processed_dict, labels_dict, label_encoder = LoadData()
    
    text = 'this is a pen'
    print(ConvertSentenceToIndices(text.split(), label_encoder))
    # [107, 18, 15, 8462]

    id_sequence_set = ConvertDatasetToIDSeqSet(corpora_processed_dict, label_encoder, torch_tensor=False)
    print(id_sequence_set['train'][:5])
    """
    [[142, 207, 3231, 0, 556, 2349, 1262, 0, 623, 0, 1534], 
    [6031, 301, 1, 493, 7478, 7444, 2360, 0, 492], 
    [12, 4, 4694, 817, 80, 2746, 35, 617, 1455, 357], 
    [411, 404, 411, 404, 49, 55, 0, 759, 19, 480, 550, 3668], 
    [869, 862, 2144, 5, 962, 4405, 0, 7185]]
    """
