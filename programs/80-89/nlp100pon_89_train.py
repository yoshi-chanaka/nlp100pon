from nlp100pon_89 import LoadTensorDatasetForBERTModel, BertClassifier

from transformers import BertModel
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

if __name__ == "__main__":
    """
    参考: https://towardsdatascience.com/text-classification-with-bert-in-pytorch-887965e5820f
    """

    dataset = LoadTensorDatasetForBERTModel()
    
    batch_size = 128
    train_dataloader    = DataLoader(dataset['train'], batch_size=batch_size, shuffle=True)
    valid_dataloader    = DataLoader(dataset['valid'], batch_size=batch_size, shuffle=False)
    test_dataloader     = DataLoader(dataset['test'], batch_size=batch_size, shuffle=False)

    train_size = len(train_dataloader.dataset)
    valid_size = len(valid_dataloader.dataset)
    test_size = len(test_dataloader.dataset)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using {} device'.format(device))
    model = BertClassifier().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-5)
    epochs = 20

    for epoch in range(epochs):
        
        model.train()
        train_loss, train_num_correct = 0, 0
        for batch in tqdm(train_dataloader):
            
            x = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            y = batch['labels'].to(device)
            pred = model.forward(x, mask)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_size
            train_num_correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        
        train_loss /= train_size
        train_acc = train_num_correct / train_size

        model.eval()
        valid_loss, valid_num_correct = 0, 0
        for batch in valid_dataloader:
            
            x = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            y = batch['labels'].to(device)
            pred = model.forward(x, mask)
            valid_loss += loss_fn(pred, y).item() * batch_size
            valid_num_correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        
        valid_loss /= valid_size
        valid_acc = valid_num_correct / valid_size

        print('epoch: {} Train [loss: {:.4f}, accuracy: {:.4f}], Valid [loss: {:.4f}, accuracy: {:.4f}]'.
                format(epoch + 1, train_loss, train_acc, valid_loss, valid_acc))


    model.eval()
    train_loss, train_num_correct = 0, 0
    for batch in train_dataloader:
        
        x = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        y = batch['labels'].to(device)
        pred = model.forward(x, mask)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * batch_size
        train_num_correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    train_loss /= train_size
    train_acc = train_num_correct / train_size


    valid_loss, valid_num_correct = 0, 0
    for batch in valid_dataloader:
        
        x = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        y = batch['labels'].to(device)
        pred = model.forward(x, mask)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        valid_loss += loss.item() * batch_size
        valid_num_correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    valid_loss /= valid_size
    valid_acc = valid_num_correct / valid_size

    
    test_loss, test_num_correct = 0, 0
    for batch in test_dataloader:
        
        x = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        y = batch['labels'].to(device)
        pred = model.forward(x, mask)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        test_loss += loss.item() * batch_size
        test_num_correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= test_size
    test_acc = test_num_correct / test_size

    print('\n----------------')
    print('Train\t[loss: {:.4f}, accuracy: {:.4f}]'.format(train_loss, train_acc))
    print('Valid\t[loss: {:.4f}, accuracy: {:.4f}]'.format(valid_loss, valid_acc))
    print('Test\t[loss: {:.4f}, accuracy: {:.4f}]'.format(test_loss, test_acc))

"""
Using cuda device
Some weights of the model checkpoint at bert-base-cased were not used when initializing BertModel: ['cls.predictions.transform.dense.weight', 'cls.predictions.transform.LayerNorm.weight', 'cls.predictions.transform.LayerNorm.bias', 'cls.seq_relationship.bias', 'cls.seq_relationship.weight', 'cls.predictions.bias', 'cls.predictions.decoder.weight', 'cls.predictions.transform.dense.bias']
- This IS expected if you are initializing BertModel from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing BertModel from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
100%|█████████████████████████████████████████████████████████████████████████| 84/84 [00:25<00:00,  3.26it/s]
epoch: 1 Train [loss: 1.1437, accuracy: 0.6265], Valid [loss: 1.0044, accuracy: 0.7924]
100%|█████████████████████████████████████████████████████████████████████████| 84/84 [00:25<00:00,  3.26it/s]
epoch: 2 Train [loss: 0.9330, accuracy: 0.8232], Valid [loss: 0.9147, accuracy: 0.8936]
100%|█████████████████████████████████████████████████████████████████████████| 84/84 [00:25<00:00,  3.25it/s]
epoch: 3 Train [loss: 0.8580, accuracy: 0.9001], Valid [loss: 0.9000, accuracy: 0.8943]
100%|█████████████████████████████████████████████████████████████████████████| 84/84 [00:25<00:00,  3.24it/s]
epoch: 4 Train [loss: 0.8319, accuracy: 0.9216], Valid [loss: 0.8796, accuracy: 0.9130]
100%|█████████████████████████████████████████████████████████████████████████| 84/84 [00:25<00:00,  3.23it/s]
epoch: 5 Train [loss: 0.8164, accuracy: 0.9355], Valid [loss: 0.8681, accuracy: 0.9280]
100%|█████████████████████████████████████████████████████████████████████████| 84/84 [00:26<00:00,  3.23it/s]
epoch: 6 Train [loss: 0.8062, accuracy: 0.9453], Valid [loss: 0.8647, accuracy: 0.9310]
100%|█████████████████████████████████████████████████████████████████████████| 84/84 [00:26<00:00,  3.22it/s]
epoch: 7 Train [loss: 0.7998, accuracy: 0.9501], Valid [loss: 0.8684, accuracy: 0.9228]
100%|█████████████████████████████████████████████████████████████████████████| 84/84 [00:26<00:00,  3.21it/s]
epoch: 8 Train [loss: 0.7967, accuracy: 0.9542], Valid [loss: 0.8717, accuracy: 0.9190]
100%|█████████████████████████████████████████████████████████████████████████| 84/84 [00:26<00:00,  3.21it/s]
epoch: 9 Train [loss: 0.7940, accuracy: 0.9563], Valid [loss: 0.8647, accuracy: 0.9288]
100%|█████████████████████████████████████████████████████████████████████████| 84/84 [00:26<00:00,  3.21it/s]
epoch: 10 Train [loss: 0.7898, accuracy: 0.9602], Valid [loss: 0.8673, accuracy: 0.9243]
100%|█████████████████████████████████████████████████████████████████████████| 84/84 [00:26<00:00,  3.21it/s]
epoch: 11 Train [loss: 0.7894, accuracy: 0.9616], Valid [loss: 0.8615, accuracy: 0.9295]
100%|█████████████████████████████████████████████████████████████████████████| 84/84 [00:26<00:00,  3.21it/s]
epoch: 12 Train [loss: 0.7851, accuracy: 0.9647], Valid [loss: 0.8663, accuracy: 0.9250]
100%|█████████████████████████████████████████████████████████████████████████| 84/84 [00:26<00:00,  3.21it/s]
epoch: 13 Train [loss: 0.7836, accuracy: 0.9659], Valid [loss: 0.8590, accuracy: 0.9318]
100%|█████████████████████████████████████████████████████████████████████████| 84/84 [00:26<00:00,  3.20it/s]
epoch: 14 Train [loss: 0.7833, accuracy: 0.9674], Valid [loss: 0.8702, accuracy: 0.9235]
100%|█████████████████████████████████████████████████████████████████████████| 84/84 [00:26<00:00,  3.20it/s]
epoch: 15 Train [loss: 0.7824, accuracy: 0.9675], Valid [loss: 0.8625, accuracy: 0.9288]
100%|█████████████████████████████████████████████████████████████████████████| 84/84 [00:26<00:00,  3.19it/s]
epoch: 16 Train [loss: 0.7810, accuracy: 0.9686], Valid [loss: 0.8619, accuracy: 0.9295]
100%|█████████████████████████████████████████████████████████████████████████| 84/84 [00:26<00:00,  3.19it/s]
epoch: 17 Train [loss: 0.7775, accuracy: 0.9721], Valid [loss: 0.8600, accuracy: 0.9303]
100%|█████████████████████████████████████████████████████████████████████████| 84/84 [00:26<00:00,  3.18it/s]
epoch: 18 Train [loss: 0.7766, accuracy: 0.9731], Valid [loss: 0.8626, accuracy: 0.9288]
100%|█████████████████████████████████████████████████████████████████████████| 84/84 [00:26<00:00,  3.20it/s]
epoch: 19 Train [loss: 0.7772, accuracy: 0.9729], Valid [loss: 0.8581, accuracy: 0.9333]
100%|█████████████████████████████████████████████████████████████████████████| 84/84 [00:26<00:00,  3.20it/s]
epoch: 20 Train [loss: 0.7778, accuracy: 0.9718], Valid [loss: 0.8609, accuracy: 0.9318]

----------------
Train   [loss: 0.7732, accuracy: 0.9761]
Valid   [loss: 0.8561, accuracy: 0.9348]
Test    [loss: 0.8734, accuracy: 0.9168]
"""

