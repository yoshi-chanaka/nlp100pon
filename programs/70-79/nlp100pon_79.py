from nlp100pon_70 import LoadData
from nlp100pon_71 import MLP
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

class NeuralNet(nn.Module):
    
    def __init__(self, in_dim, hid1_dim, hid2_dim, out_dim):
        super(NeuralNet, self).__init__()
        self.linear1 = nn.Linear(in_dim, hid1_dim)
        self.linear2 = nn.Linear(hid1_dim, hid2_dim)
        self.linear3 = nn.Linear(hid2_dim, out_dim)

        self.bn1 = nn.BatchNorm1d(hid1_dim)
        self.bn2 = nn.BatchNorm1d(hid2_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = self.bn1(x).relu()
        x = self.linear2(x)
        x = self.bn2(x).relu()
        x = self.linear3(x).relu()
        return F.softmax(x, dim=1)

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

    hid1_dim, hid2_dim = 128, 32
    model = NeuralNet(in_dim=300, hid1_dim=hid1_dim, hid2_dim=hid2_dim, out_dim=4).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.5)
    epochs = 100

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
        
    model.eval()
    
    loss_fn = nn.CrossEntropyLoss()
    
    pred = model.forward(X_train.to(device)).cpu()
    train_loss = loss_fn(pred, y_train)
    train_acc = (pred.argmax(1) == y_train).type(torch.float).sum().item() / train_size

    pred = model.forward(X_valid.to(device)).cpu()
    valid_loss = loss_fn(pred, y_valid)
    valid_acc = (pred.argmax(1) == y_valid).type(torch.float).sum().item() / valid_size

    pred = model.forward(X_test.to(device)).cpu()
    test_loss = loss_fn(pred, y_test)
    test_acc = (pred.argmax(1) == y_test).type(torch.float).sum().item() / test_size

    print('\n----------------')
    print('Train\t[loss: {:.4f}, accuracy: {:.4f}]'.format(train_loss, train_acc))
    print('Valid\t[loss: {:.4f}, accuracy: {:.4f}]'.format(valid_loss, valid_acc))
    print('Test\t[loss: {:.4f}, accuracy: {:.4f}]'.format(test_loss, test_acc))

"""
Using cuda device
epoch: 1 Train [loss: 0.9395, accuracy: 0.8204], Valid [loss: 0.8694, accuracy: 0.8943]
epoch: 2 Train [loss: 0.8578, accuracy: 0.8911], Valid [loss: 0.8562, accuracy: 0.8988]
epoch: 3 Train [loss: 0.8418, accuracy: 0.9068], Valid [loss: 0.8488, accuracy: 0.9025]
epoch: 4 Train [loss: 0.8285, accuracy: 0.9196], Valid [loss: 0.8474, accuracy: 0.9048]
epoch: 5 Train [loss: 0.8186, accuracy: 0.9298], Valid [loss: 0.8469, accuracy: 0.9003]
epoch: 6 Train [loss: 0.8131, accuracy: 0.9339], Valid [loss: 0.8426, accuracy: 0.9093]
epoch: 7 Train [loss: 0.8075, accuracy: 0.9394], Valid [loss: 0.8420, accuracy: 0.9063]
epoch: 8 Train [loss: 0.8040, accuracy: 0.9427], Valid [loss: 0.8439, accuracy: 0.9070]
epoch: 9 Train [loss: 0.7970, accuracy: 0.9506], Valid [loss: 0.8442, accuracy: 0.9070]
epoch: 10 Train [loss: 0.7932, accuracy: 0.9532], Valid [loss: 0.8448, accuracy: 0.9070]
epoch: 11 Train [loss: 0.7889, accuracy: 0.9582], Valid [loss: 0.8474, accuracy: 0.9010]
epoch: 12 Train [loss: 0.7887, accuracy: 0.9576], Valid [loss: 0.8423, accuracy: 0.9070]
epoch: 13 Train [loss: 0.7851, accuracy: 0.9611], Valid [loss: 0.8499, accuracy: 0.8958]
epoch: 14 Train [loss: 0.7839, accuracy: 0.9624], Valid [loss: 0.8465, accuracy: 0.9033]
epoch: 15 Train [loss: 0.7815, accuracy: 0.9646], Valid [loss: 0.8391, accuracy: 0.9100]
epoch: 16 Train [loss: 0.7813, accuracy: 0.9644], Valid [loss: 0.8491, accuracy: 0.8981]
epoch: 17 Train [loss: 0.7802, accuracy: 0.9659], Valid [loss: 0.8437, accuracy: 0.9063]
epoch: 18 Train [loss: 0.7794, accuracy: 0.9664], Valid [loss: 0.8480, accuracy: 0.8973]
epoch: 19 Train [loss: 0.7765, accuracy: 0.9688], Valid [loss: 0.8444, accuracy: 0.9078]
epoch: 20 Train [loss: 0.7735, accuracy: 0.9728], Valid [loss: 0.8459, accuracy: 0.9018]
epoch: 21 Train [loss: 0.7744, accuracy: 0.9712], Valid [loss: 0.8427, accuracy: 0.9085]
epoch: 22 Train [loss: 0.7734, accuracy: 0.9727], Valid [loss: 0.8442, accuracy: 0.9033]
epoch: 23 Train [loss: 0.7743, accuracy: 0.9711], Valid [loss: 0.8442, accuracy: 0.9055]
epoch: 24 Train [loss: 0.7715, accuracy: 0.9745], Valid [loss: 0.8465, accuracy: 0.9018]
epoch: 25 Train [loss: 0.7707, accuracy: 0.9754], Valid [loss: 0.8413, accuracy: 0.9063]
epoch: 26 Train [loss: 0.7722, accuracy: 0.9732], Valid [loss: 0.8577, accuracy: 0.8891]
epoch: 27 Train [loss: 0.7724, accuracy: 0.9733], Valid [loss: 0.8507, accuracy: 0.8988]
epoch: 28 Train [loss: 0.7699, accuracy: 0.9758], Valid [loss: 0.8479, accuracy: 0.8988]
epoch: 29 Train [loss: 0.7702, accuracy: 0.9756], Valid [loss: 0.8470, accuracy: 0.9033]
epoch: 30 Train [loss: 0.7703, accuracy: 0.9753], Valid [loss: 0.8521, accuracy: 0.8988]
epoch: 31 Train [loss: 0.7698, accuracy: 0.9756], Valid [loss: 0.8422, accuracy: 0.9085]
epoch: 32 Train [loss: 0.7688, accuracy: 0.9768], Valid [loss: 0.8393, accuracy: 0.9115]
epoch: 33 Train [loss: 0.7679, accuracy: 0.9775], Valid [loss: 0.8410, accuracy: 0.9063]
epoch: 34 Train [loss: 0.7685, accuracy: 0.9769], Valid [loss: 0.8433, accuracy: 0.9048]
epoch: 35 Train [loss: 0.7690, accuracy: 0.9766], Valid [loss: 0.8400, accuracy: 0.9040]
epoch: 36 Train [loss: 0.7683, accuracy: 0.9765], Valid [loss: 0.8399, accuracy: 0.9115]
epoch: 37 Train [loss: 0.7665, accuracy: 0.9790], Valid [loss: 0.8482, accuracy: 0.9010]
epoch: 38 Train [loss: 0.7649, accuracy: 0.9806], Valid [loss: 0.8448, accuracy: 0.9040]
epoch: 39 Train [loss: 0.7665, accuracy: 0.9783], Valid [loss: 0.8423, accuracy: 0.9048]
epoch: 40 Train [loss: 0.7668, accuracy: 0.9784], Valid [loss: 0.8400, accuracy: 0.9093]
epoch: 41 Train [loss: 0.7654, accuracy: 0.9798], Valid [loss: 0.8434, accuracy: 0.9033]
epoch: 42 Train [loss: 0.7651, accuracy: 0.9801], Valid [loss: 0.8436, accuracy: 0.9048]
epoch: 43 Train [loss: 0.7645, accuracy: 0.9809], Valid [loss: 0.8427, accuracy: 0.9025]
epoch: 44 Train [loss: 0.7644, accuracy: 0.9810], Valid [loss: 0.8359, accuracy: 0.9145]
epoch: 45 Train [loss: 0.7656, accuracy: 0.9793], Valid [loss: 0.8466, accuracy: 0.9025]
epoch: 46 Train [loss: 0.7666, accuracy: 0.9791], Valid [loss: 0.8409, accuracy: 0.9078]
epoch: 47 Train [loss: 0.7649, accuracy: 0.9806], Valid [loss: 0.8474, accuracy: 0.8996]
epoch: 48 Train [loss: 0.7632, accuracy: 0.9822], Valid [loss: 0.8462, accuracy: 0.9033]
epoch: 49 Train [loss: 0.7645, accuracy: 0.9805], Valid [loss: 0.8430, accuracy: 0.9048]
epoch: 50 Train [loss: 0.7637, accuracy: 0.9815], Valid [loss: 0.8411, accuracy: 0.9055]
epoch: 51 Train [loss: 0.7632, accuracy: 0.9818], Valid [loss: 0.8406, accuracy: 0.9100]
epoch: 52 Train [loss: 0.7626, accuracy: 0.9826], Valid [loss: 0.8431, accuracy: 0.9040]
epoch: 53 Train [loss: 0.7623, accuracy: 0.9830], Valid [loss: 0.8391, accuracy: 0.9085]
epoch: 54 Train [loss: 0.7617, accuracy: 0.9836], Valid [loss: 0.8383, accuracy: 0.9040]
epoch: 55 Train [loss: 0.7620, accuracy: 0.9831], Valid [loss: 0.8392, accuracy: 0.9130]
epoch: 56 Train [loss: 0.7617, accuracy: 0.9831], Valid [loss: 0.8392, accuracy: 0.9078]
epoch: 57 Train [loss: 0.7614, accuracy: 0.9835], Valid [loss: 0.8424, accuracy: 0.9063]
epoch: 58 Train [loss: 0.7622, accuracy: 0.9829], Valid [loss: 0.8437, accuracy: 0.9025]
epoch: 59 Train [loss: 0.7629, accuracy: 0.9823], Valid [loss: 0.8390, accuracy: 0.9108]
epoch: 60 Train [loss: 0.7615, accuracy: 0.9836], Valid [loss: 0.8431, accuracy: 0.9025]
epoch: 61 Train [loss: 0.7608, accuracy: 0.9842], Valid [loss: 0.8463, accuracy: 0.8996]
epoch: 62 Train [loss: 0.7607, accuracy: 0.9844], Valid [loss: 0.8431, accuracy: 0.9040]
epoch: 63 Train [loss: 0.7606, accuracy: 0.9844], Valid [loss: 0.8444, accuracy: 0.9033]
epoch: 64 Train [loss: 0.7599, accuracy: 0.9851], Valid [loss: 0.8421, accuracy: 0.9018]
epoch: 65 Train [loss: 0.7610, accuracy: 0.9839], Valid [loss: 0.8441, accuracy: 0.9040]
epoch: 66 Train [loss: 0.7609, accuracy: 0.9842], Valid [loss: 0.8416, accuracy: 0.9063]
epoch: 67 Train [loss: 0.7609, accuracy: 0.9841], Valid [loss: 0.8408, accuracy: 0.9093]
epoch: 68 Train [loss: 0.7594, accuracy: 0.9856], Valid [loss: 0.8409, accuracy: 0.9070]
epoch: 69 Train [loss: 0.7598, accuracy: 0.9854], Valid [loss: 0.8421, accuracy: 0.9048]
epoch: 70 Train [loss: 0.7593, accuracy: 0.9857], Valid [loss: 0.8412, accuracy: 0.9085]
epoch: 71 Train [loss: 0.7597, accuracy: 0.9850], Valid [loss: 0.8386, accuracy: 0.9130]
epoch: 72 Train [loss: 0.7599, accuracy: 0.9851], Valid [loss: 0.8387, accuracy: 0.9100]
epoch: 73 Train [loss: 0.7605, accuracy: 0.9846], Valid [loss: 0.8383, accuracy: 0.9100]
epoch: 74 Train [loss: 0.7607, accuracy: 0.9843], Valid [loss: 0.8393, accuracy: 0.9085]
epoch: 75 Train [loss: 0.7600, accuracy: 0.9853], Valid [loss: 0.8408, accuracy: 0.9063]
epoch: 76 Train [loss: 0.7599, accuracy: 0.9852], Valid [loss: 0.8434, accuracy: 0.9048]
epoch: 77 Train [loss: 0.7599, accuracy: 0.9851], Valid [loss: 0.8387, accuracy: 0.9115]
epoch: 78 Train [loss: 0.7598, accuracy: 0.9853], Valid [loss: 0.8404, accuracy: 0.9100]
epoch: 79 Train [loss: 0.7591, accuracy: 0.9859], Valid [loss: 0.8433, accuracy: 0.9048]
epoch: 80 Train [loss: 0.7590, accuracy: 0.9859], Valid [loss: 0.8439, accuracy: 0.9040]
epoch: 81 Train [loss: 0.7590, accuracy: 0.9862], Valid [loss: 0.8486, accuracy: 0.8988]
epoch: 82 Train [loss: 0.7597, accuracy: 0.9853], Valid [loss: 0.8365, accuracy: 0.9123]
epoch: 83 Train [loss: 0.7590, accuracy: 0.9859], Valid [loss: 0.8405, accuracy: 0.9070]
epoch: 84 Train [loss: 0.7592, accuracy: 0.9859], Valid [loss: 0.8452, accuracy: 0.9033]
epoch: 85 Train [loss: 0.7587, accuracy: 0.9863], Valid [loss: 0.8636, accuracy: 0.8831]
epoch: 86 Train [loss: 0.7595, accuracy: 0.9854], Valid [loss: 0.8428, accuracy: 0.9055]
epoch: 87 Train [loss: 0.7590, accuracy: 0.9859], Valid [loss: 0.8375, accuracy: 0.9108]
epoch: 88 Train [loss: 0.7592, accuracy: 0.9859], Valid [loss: 0.8411, accuracy: 0.9063]
epoch: 89 Train [loss: 0.7599, accuracy: 0.9853], Valid [loss: 0.8462, accuracy: 0.9033]
epoch: 90 Train [loss: 0.7604, accuracy: 0.9844], Valid [loss: 0.8414, accuracy: 0.9055]
epoch: 91 Train [loss: 0.7598, accuracy: 0.9853], Valid [loss: 0.8426, accuracy: 0.9055]
epoch: 92 Train [loss: 0.7590, accuracy: 0.9860], Valid [loss: 0.8419, accuracy: 0.9093]
epoch: 93 Train [loss: 0.7595, accuracy: 0.9853], Valid [loss: 0.8383, accuracy: 0.9100]
epoch: 94 Train [loss: 0.7591, accuracy: 0.9859], Valid [loss: 0.8396, accuracy: 0.9070]
epoch: 95 Train [loss: 0.7588, accuracy: 0.9860], Valid [loss: 0.8376, accuracy: 0.9115]
epoch: 96 Train [loss: 0.7592, accuracy: 0.9858], Valid [loss: 0.8397, accuracy: 0.9100]
epoch: 97 Train [loss: 0.7590, accuracy: 0.9860], Valid [loss: 0.8399, accuracy: 0.9085]
epoch: 98 Train [loss: 0.7586, accuracy: 0.9863], Valid [loss: 0.8380, accuracy: 0.9108]
epoch: 99 Train [loss: 0.7586, accuracy: 0.9865], Valid [loss: 0.8379, accuracy: 0.9130]
epoch: 100 Train [loss: 0.7589, accuracy: 0.9859], Valid [loss: 0.8368, accuracy: 0.9130]

----------------
Train   [loss: 0.7565, accuracy: 0.9872]
Valid   [loss: 0.8300, accuracy: 0.9130]
Test    [loss: 0.8225, accuracy: 0.9213]
"""