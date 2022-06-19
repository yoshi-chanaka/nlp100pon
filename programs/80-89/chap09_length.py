import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from nlp100pon_80 import ConvertDatasetToIDSeqSet, LoadData
import torch

if __name__ == "__main__":
    
    corpora_processed_dict, labels_dict, label_encoder = LoadData()
    id_sequence_set = ConvertDatasetToIDSeqSet(corpora_processed_dict, label_encoder)

    length_list = []
    for k in id_sequence_set.keys():
        length_list += [len(seq) for seq in id_sequence_set[k]]

    plt.boxplot(length_list)
    plt.savefig('../../figures/chap09_length.jpg')
    plt.show()
