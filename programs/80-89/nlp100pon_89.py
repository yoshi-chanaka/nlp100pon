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
        break

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
token_type_ids: torch.Size([8, 30])
attention_mask: torch.Size([8, 30])
labels: torch.Size([8])
Using cuda device
Some weights of the model checkpoint at bert-base-cased were not used when initializing BertModel: ['cls.seq_relationship.bias', 'cls.predictions.bias', 'cls.predictions.transform.LayerNorm.bias', 'cls.seq_relationship.weight', 'cls.predictions.transform.dense.weight', 'cls.predictions.transform.LayerNorm.weight', 'cls.predictions.decoder.weight', 'cls.predictions.transform.dense.bias']
- This IS expected if you are initializing BertModel from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing BertModel from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
tensor([[0.1695, 0.3256, 0.3063, 0.1985],
        [0.1762, 0.2991, 0.3185, 0.2063],
        [0.1778, 0.3274, 0.2856, 0.2092],
        [0.1549, 0.3094, 0.3226, 0.2131],
        [0.1662, 0.4184, 0.2647, 0.1506],
        [0.1485, 0.2712, 0.3847, 0.1956],
        [0.1564, 0.3046, 0.3233, 0.2157],
        [0.2148, 0.3079, 0.2906, 0.1867]], device='cuda:0',
       grad_fn=<SliceBackward0>)
"""