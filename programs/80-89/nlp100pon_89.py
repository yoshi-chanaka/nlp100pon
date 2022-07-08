from nlp100pon_80 import ConvertDatasetToIDSeqSet, LoadData
from transformers import BertModel
from torch.utils.data import DataLoader

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

def LoadTensorDatasetForBERTModel(max_length=30):

    from transformers import BertTokenizer
    from torch.utils.data import TensorDataset
    from nlp100pon_80 import ConvertDatasetToIDSeqSet, LoadData

    
    corpora_processed_dict, labels_dict, _ = LoadData()
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

    dataset = {}
    for k in corpora_processed_dict.keys():
        # k ∈ {'train', 'valid', 'test'}
        dataset[k] = []
        
        for i, text in enumerate(corpora_processed_dict[k]):
            enc = tokenizer(text,
                            max_length=max_length,
                            padding='max_length',
                            truncation=True)
            enc['labels'] = labels_dict[k][i].item()
            enc['input_ids'] = torch.tensor(enc['input_ids'])
            enc['token_type_ids'] = torch.tensor(enc['token_type_ids'])
            enc['attention_mask'] = torch.tensor(enc['attention_mask'])
            dataset[k].append(enc)

    return dataset
    

class BertClassifier(nn.Module):

    def __init__(self, dropout_rate=0.1):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(768, 4)

    def forward(self, x, mask):
        _, x = self.bert(input_ids= x, attention_mask=mask, return_dict=False)
        x = self.dropout(x)
        x = self.linear(x)
        
        return F.softmax(x, dim=1)


if __name__ == "__main__":

    dataset = LoadTensorDatasetForBERTModel()
    
    batch_size = 8
    train_dataloader    = DataLoader(dataset['train'], batch_size=batch_size, shuffle=True)
    valid_dataloader    = DataLoader(dataset['valid'], batch_size=batch_size, shuffle=False)
    test_dataloader     = DataLoader(dataset['test'], batch_size=batch_size, shuffle=False)

    train_size = len(train_dataloader.dataset)
    valid_size = len(valid_dataloader.dataset)
    test_size = len(test_dataloader.dataset)

    for batch in train_dataloader:
        print('batch.keys: {}'.format(batch.keys()))
        for k in batch.keys():
            print('{}:\t{}'.format(k, batch[k].shape))
            print(batch[k][0])
        break
    
    print('=' * 30)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using {} device'.format(device))
    
    model = BertClassifier().to(device)
    for batch in train_dataloader:
        y = model.forward(batch['input_ids'].to(device), batch['attention_mask'].to(device))
        print(y[:10])
        break

"""
batch.keys: dict_keys(['input_ids', 'token_type_ids', 'attention_mask', 'labels'])
input_ids:      torch.Size([8, 30])
tensor([  101, 15117,  7352,  2303,  1106,   109,   124,   127,   129,   128,
          127,   170, 20003,  4934,  5980,  1867,   102,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0])
token_type_ids: torch.Size([8, 30])
tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0])
attention_mask: torch.Size([8, 30])
tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0])
labels: torch.Size([8])
tensor(1)
==============================
Using cuda device
Some weights of the model checkpoint at bert-base-cased were not used when initializing BertModel: ['cls.predictions.transform.LayerNorm.bias', 'cls.predictions.transform.dense.bias', 'cls.predictions.decoder.weight', 'cls.seq_relationship.weight', 'cls.predictions.transform.LayerNorm.weight', 'cls.predictions.bias', 'cls.predictions.transform.dense.weight', 'cls.seq_relationship.bias']
- This IS expected if you are initializing BertModel from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing BertModel from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
tensor([[0.1858, 0.2973, 0.2854, 0.2315],
        [0.2023, 0.2813, 0.2714, 0.2450],
        [0.1365, 0.3335, 0.2738, 0.2562],
        [0.1722, 0.3546, 0.2588, 0.2143],
        [0.1334, 0.2557, 0.2958, 0.3150],
        [0.1530, 0.3122, 0.2862, 0.2485],
        [0.1448, 0.2399, 0.4153, 0.2000],
        [0.2289, 0.2386, 0.2700, 0.2625]], device='cuda:0',
       grad_fn=<SliceBackward0>)
"""