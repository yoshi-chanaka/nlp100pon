from nlp100pon_70 import LoadData
from nlp100pon_71 import MLP
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

if __name__ == "__main__":

    path = '../../data/NewsAggregatorDataset/chap08_avgembed.pickle'
    X, Y = LoadData(path)    
    X_train, y_train    = torch.from_numpy(X['train'].astype(np.float32)), torch.tensor(Y['train'])
    X_valid, y_valid    = torch.from_numpy(X['valid'].astype(np.float32)), torch.tensor(Y['valid'])
    X_test, y_test      = torch.from_numpy(X['test'].astype(np.float32)), torch.tensor(Y['test'])

    dataset = {}
    dataset['train']    = torch.utils.data.TensorDataset(X_train, y_train)
    dataset['valid']    = torch.utils.data.TensorDataset(X_valid, y_valid)
    dataset['test']     = torch.utils.data.TensorDataset(X_test, y_test)


    batch_size = 64
    train_dataloader    = DataLoader(dataset['train'], batch_size=batch_size, shuffle=True)
    valid_dataloader    = DataLoader(dataset['valid'], batch_size=batch_size, shuffle=False)
    test_dataloader     = DataLoader(dataset['test'], batch_size=batch_size, shuffle=False)
    train_size, valid_size, test_size = len(train_dataloader.dataset), len(valid_dataloader.dataset), len(test_dataloader.dataset)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} device'.format(device))

    model = MLP(in_dim=300, out_dim=4).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.99)
    epochs = 100

    history_train_loss, history_valid_loss = [], []
    history_train_acc, history_valid_acc = [], []
    checkpoints = [] # 問76 チェックポイント

    for epoch in range(epochs):
        
        model.train()
        train_loss, train_num_correct = 0, 0
        for X, y in train_dataloader:

            X, y = X.to(device), y.to(device)
            pred = model(X)
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
        for X, y in valid_dataloader:

            X, y = X.to(device), y.to(device)
            pred = model(X)
            valid_loss += loss_fn(pred, y).item() * batch_size
            valid_num_correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        
        valid_loss /= valid_size
        valid_acc = valid_num_correct / valid_size

        print('epoch: {} Train [loss: {:.4f}, accuracy: {:.4f}], Valid [loss: {:.4f}, accuracy: {:.4f}]'.
                format(epoch + 1, train_loss, train_acc, valid_loss, valid_acc))
            
        history_train_loss.append(train_loss)
        history_valid_loss.append(valid_loss)
        history_train_acc.append(train_acc)
        history_valid_acc.append(valid_acc)

        checkpoints.append({
                        'epoch': epoch,
                        'model_state': model.state_dict(),
                        'optimizer_state': optimizer.state_dict(),
                        'train_loss': train_loss,
                        'valid_loss': valid_loss
                    })
    
    x_ = np.arange(1, epochs + 1)

    plt.figure()
    plt.plot(x_, history_train_loss, label='train')
    plt.plot(x_, history_valid_loss, label='valid')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig('../../figures/chap08_75_loss.jpg')

    plt.figure()
    plt.plot(x_, history_train_acc, label='train')
    plt.plot(x_, history_valid_acc, label='valid')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.savefig('../../figures/chap08_75_acc.jpg')


    # 問76. チェックポイント
    save_path = '../../materials/chap08_76_checkpoint.pt'
    torch.save(checkpoints, save_path)

    checkpoint = torch.load(save_path)
    print('\ncheckpoint[0]: ')
    print(checkpoint[0])

"""
...
epoch: 90 Train [loss: 0.8603, accuracy: 0.8979], Valid [loss: 0.8657, accuracy: 0.8951]
epoch: 91 Train [loss: 0.8601, accuracy: 0.8981], Valid [loss: 0.8656, accuracy: 0.8951]
epoch: 92 Train [loss: 0.8598, accuracy: 0.8978], Valid [loss: 0.8655, accuracy: 0.8958]
epoch: 93 Train [loss: 0.8598, accuracy: 0.8981], Valid [loss: 0.8653, accuracy: 0.8958]
epoch: 94 Train [loss: 0.8594, accuracy: 0.8981], Valid [loss: 0.8652, accuracy: 0.8966]
epoch: 95 Train [loss: 0.8592, accuracy: 0.8984], Valid [loss: 0.8651, accuracy: 0.8966]
epoch: 96 Train [loss: 0.8589, accuracy: 0.8991], Valid [loss: 0.8650, accuracy: 0.8966]
epoch: 97 Train [loss: 0.8588, accuracy: 0.8997], Valid [loss: 0.8648, accuracy: 0.8973]
epoch: 98 Train [loss: 0.8586, accuracy: 0.8988], Valid [loss: 0.8647, accuracy: 0.8981]
epoch: 99 Train [loss: 0.8584, accuracy: 0.8994], Valid [loss: 0.8646, accuracy: 0.8966]
epoch: 100 Train [loss: 0.8582, accuracy: 0.8989], Valid [loss: 0.8646, accuracy: 0.8966]

checkpoint[0]:
{'epoch': 0, 'model_state': OrderedDict([('linear.weight', tensor([[-1.3990, -0.1286, -0.1925,  ...,  0.6669,  0.5246, -1.9964],
        [ 1.6871, -0.8815,  1.3383,  ..., -0.2901, -0.4602, -1.1388],
        [-0.2371,  1.4641, -2.1109,  ..., -1.1226, -1.0566,  1.3618],
        [-0.1378, -0.4504,  1.0195,  ...,  0.7326,  0.9366,  1.7944]],
       device='cuda:0'))]), 'optimizer_state': {'state': {0: {'momentum_buffer': None}}, 'param_groups': [{'lr': 0.99, 'momentum': 0, 'dampening': 0, 'weight_decay': 0, 'nesterov': False, 'maximize': False, 'params': [0]}]}, 'train_loss': 1.106586699006797, 'valid_loss': 1.0194595457016975}
"""

