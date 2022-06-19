from nlp100pon_70 import LoadData
from nlp100pon_71 import MLP
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

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

    print('\nQ.74\n----------------')
    print('Train\t[loss: {:.4f}, accuracy: {:.4f}]'.format(train_loss, train_acc))
    print('Valid\t[loss: {:.4f}, accuracy: {:.4f}]'.format(valid_loss, valid_acc))
    print('Test\t[loss: {:.4f}, accuracy: {:.4f}]'.format(test_loss, test_acc))

"""
Using cuda device
epoch: 1 Train [loss: 1.1066, accuracy: 0.7508], Valid [loss: 1.0191, accuracy: 0.7804]
epoch: 2 Train [loss: 1.0066, accuracy: 0.7714], Valid [loss: 0.9951, accuracy: 0.7864]
epoch: 3 Train [loss: 0.9908, accuracy: 0.7760], Valid [loss: 0.9856, accuracy: 0.7879]
epoch: 4 Train [loss: 0.9833, accuracy: 0.7779], Valid [loss: 0.9801, accuracy: 0.7894]
epoch: 5 Train [loss: 0.9785, accuracy: 0.7796], Valid [loss: 0.9763, accuracy: 0.7894]
epoch: 6 Train [loss: 0.9748, accuracy: 0.7805], Valid [loss: 0.9732, accuracy: 0.7924]
epoch: 7 Train [loss: 0.9716, accuracy: 0.7819], Valid [loss: 0.9700, accuracy: 0.7924]
epoch: 8 Train [loss: 0.9678, accuracy: 0.7824], Valid [loss: 0.9656, accuracy: 0.7954]
epoch: 9 Train [loss: 0.9624, accuracy: 0.7873], Valid [loss: 0.9588, accuracy: 0.8021]
epoch: 10 Train [loss: 0.9554, accuracy: 0.8002], Valid [loss: 0.9508, accuracy: 0.8141]
epoch: 11 Train [loss: 0.9494, accuracy: 0.8095], Valid [loss: 0.9447, accuracy: 0.8246]
epoch: 12 Train [loss: 0.9451, accuracy: 0.8136], Valid [loss: 0.9405, accuracy: 0.8298]
epoch: 13 Train [loss: 0.9419, accuracy: 0.8167], Valid [loss: 0.9369, accuracy: 0.8336]
epoch: 14 Train [loss: 0.9390, accuracy: 0.8206], Valid [loss: 0.9340, accuracy: 0.8328]
epoch: 15 Train [loss: 0.9363, accuracy: 0.8216], Valid [loss: 0.9313, accuracy: 0.8373]
epoch: 16 Train [loss: 0.9337, accuracy: 0.8226], Valid [loss: 0.9286, accuracy: 0.8388]
epoch: 17 Train [loss: 0.9306, accuracy: 0.8250], Valid [loss: 0.9254, accuracy: 0.8426]
epoch: 18 Train [loss: 0.9263, accuracy: 0.8304], Valid [loss: 0.9210, accuracy: 0.8471]
epoch: 19 Train [loss: 0.9205, accuracy: 0.8415], Valid [loss: 0.9160, accuracy: 0.8606]
epoch: 20 Train [loss: 0.9151, accuracy: 0.8511], Valid [loss: 0.9118, accuracy: 0.8681]
epoch: 21 Train [loss: 0.9109, accuracy: 0.8571], Valid [loss: 0.9081, accuracy: 0.8756]
epoch: 22 Train [loss: 0.9073, accuracy: 0.8614], Valid [loss: 0.9050, accuracy: 0.8816]
epoch: 23 Train [loss: 0.9043, accuracy: 0.8626], Valid [loss: 0.9022, accuracy: 0.8823]
epoch: 24 Train [loss: 0.9016, accuracy: 0.8651], Valid [loss: 0.8997, accuracy: 0.8853]
epoch: 25 Train [loss: 0.8993, accuracy: 0.8678], Valid [loss: 0.8975, accuracy: 0.8853]
epoch: 26 Train [loss: 0.8972, accuracy: 0.8685], Valid [loss: 0.8956, accuracy: 0.8861]
epoch: 27 Train [loss: 0.8954, accuracy: 0.8705], Valid [loss: 0.8938, accuracy: 0.8861]
epoch: 28 Train [loss: 0.8937, accuracy: 0.8713], Valid [loss: 0.8922, accuracy: 0.8868]
epoch: 29 Train [loss: 0.8919, accuracy: 0.8722], Valid [loss: 0.8908, accuracy: 0.8861]
epoch: 30 Train [loss: 0.8906, accuracy: 0.8734], Valid [loss: 0.8894, accuracy: 0.8868]
epoch: 31 Train [loss: 0.8892, accuracy: 0.8742], Valid [loss: 0.8881, accuracy: 0.8883]
epoch: 32 Train [loss: 0.8881, accuracy: 0.8744], Valid [loss: 0.8870, accuracy: 0.8883]
epoch: 33 Train [loss: 0.8868, accuracy: 0.8757], Valid [loss: 0.8860, accuracy: 0.8891]
epoch: 34 Train [loss: 0.8858, accuracy: 0.8766], Valid [loss: 0.8850, accuracy: 0.8876]
epoch: 35 Train [loss: 0.8846, accuracy: 0.8775], Valid [loss: 0.8841, accuracy: 0.8868]
epoch: 36 Train [loss: 0.8837, accuracy: 0.8778], Valid [loss: 0.8833, accuracy: 0.8861]
epoch: 37 Train [loss: 0.8828, accuracy: 0.8789], Valid [loss: 0.8824, accuracy: 0.8861]
epoch: 38 Train [loss: 0.8819, accuracy: 0.8792], Valid [loss: 0.8817, accuracy: 0.8853]
epoch: 39 Train [loss: 0.8809, accuracy: 0.8803], Valid [loss: 0.8809, accuracy: 0.8853]
epoch: 40 Train [loss: 0.8802, accuracy: 0.8804], Valid [loss: 0.8802, accuracy: 0.8868]
epoch: 41 Train [loss: 0.8794, accuracy: 0.8811], Valid [loss: 0.8797, accuracy: 0.8861]
epoch: 42 Train [loss: 0.8788, accuracy: 0.8824], Valid [loss: 0.8790, accuracy: 0.8868]
epoch: 43 Train [loss: 0.8780, accuracy: 0.8827], Valid [loss: 0.8784, accuracy: 0.8868]
epoch: 44 Train [loss: 0.8773, accuracy: 0.8835], Valid [loss: 0.8779, accuracy: 0.8876]
epoch: 45 Train [loss: 0.8766, accuracy: 0.8836], Valid [loss: 0.8773, accuracy: 0.8876]
epoch: 46 Train [loss: 0.8759, accuracy: 0.8845], Valid [loss: 0.8769, accuracy: 0.8876]
epoch: 47 Train [loss: 0.8754, accuracy: 0.8852], Valid [loss: 0.8764, accuracy: 0.8876]
epoch: 48 Train [loss: 0.8748, accuracy: 0.8853], Valid [loss: 0.8759, accuracy: 0.8898]
epoch: 49 Train [loss: 0.8744, accuracy: 0.8860], Valid [loss: 0.8755, accuracy: 0.8906]
epoch: 50 Train [loss: 0.8737, accuracy: 0.8863], Valid [loss: 0.8751, accuracy: 0.8906]
epoch: 51 Train [loss: 0.8731, accuracy: 0.8869], Valid [loss: 0.8746, accuracy: 0.8906]
epoch: 52 Train [loss: 0.8727, accuracy: 0.8871], Valid [loss: 0.8742, accuracy: 0.8921]
epoch: 53 Train [loss: 0.8722, accuracy: 0.8872], Valid [loss: 0.8739, accuracy: 0.8921]
epoch: 54 Train [loss: 0.8717, accuracy: 0.8880], Valid [loss: 0.8735, accuracy: 0.8906]
epoch: 55 Train [loss: 0.8712, accuracy: 0.8887], Valid [loss: 0.8731, accuracy: 0.8906]
epoch: 56 Train [loss: 0.8708, accuracy: 0.8891], Valid [loss: 0.8728, accuracy: 0.8898]
epoch: 57 Train [loss: 0.8704, accuracy: 0.8895], Valid [loss: 0.8724, accuracy: 0.8898]
epoch: 58 Train [loss: 0.8699, accuracy: 0.8895], Valid [loss: 0.8721, accuracy: 0.8891]
epoch: 59 Train [loss: 0.8695, accuracy: 0.8898], Valid [loss: 0.8719, accuracy: 0.8891]
epoch: 60 Train [loss: 0.8691, accuracy: 0.8905], Valid [loss: 0.8716, accuracy: 0.8891]
epoch: 61 Train [loss: 0.8686, accuracy: 0.8906], Valid [loss: 0.8713, accuracy: 0.8891]
epoch: 62 Train [loss: 0.8683, accuracy: 0.8904], Valid [loss: 0.8710, accuracy: 0.8891]
epoch: 63 Train [loss: 0.8679, accuracy: 0.8911], Valid [loss: 0.8708, accuracy: 0.8891]
epoch: 64 Train [loss: 0.8676, accuracy: 0.8916], Valid [loss: 0.8705, accuracy: 0.8891]
epoch: 65 Train [loss: 0.8672, accuracy: 0.8917], Valid [loss: 0.8703, accuracy: 0.8891]
epoch: 66 Train [loss: 0.8668, accuracy: 0.8921], Valid [loss: 0.8699, accuracy: 0.8898]
epoch: 67 Train [loss: 0.8665, accuracy: 0.8927], Valid [loss: 0.8697, accuracy: 0.8898]
epoch: 68 Train [loss: 0.8661, accuracy: 0.8929], Valid [loss: 0.8694, accuracy: 0.8906]
epoch: 69 Train [loss: 0.8658, accuracy: 0.8930], Valid [loss: 0.8692, accuracy: 0.8913]
epoch: 70 Train [loss: 0.8654, accuracy: 0.8933], Valid [loss: 0.8691, accuracy: 0.8913]
epoch: 71 Train [loss: 0.8652, accuracy: 0.8940], Valid [loss: 0.8687, accuracy: 0.8928]
epoch: 72 Train [loss: 0.8649, accuracy: 0.8939], Valid [loss: 0.8686, accuracy: 0.8928]
epoch: 73 Train [loss: 0.8646, accuracy: 0.8941], Valid [loss: 0.8684, accuracy: 0.8921]
epoch: 74 Train [loss: 0.8643, accuracy: 0.8944], Valid [loss: 0.8682, accuracy: 0.8921]
epoch: 75 Train [loss: 0.8640, accuracy: 0.8945], Valid [loss: 0.8680, accuracy: 0.8928]
epoch: 76 Train [loss: 0.8638, accuracy: 0.8947], Valid [loss: 0.8678, accuracy: 0.8921]
epoch: 77 Train [loss: 0.8634, accuracy: 0.8949], Valid [loss: 0.8676, accuracy: 0.8921]
epoch: 78 Train [loss: 0.8631, accuracy: 0.8951], Valid [loss: 0.8675, accuracy: 0.8921]
epoch: 79 Train [loss: 0.8630, accuracy: 0.8952], Valid [loss: 0.8673, accuracy: 0.8921]
epoch: 80 Train [loss: 0.8627, accuracy: 0.8958], Valid [loss: 0.8671, accuracy: 0.8921]
epoch: 81 Train [loss: 0.8623, accuracy: 0.8952], Valid [loss: 0.8669, accuracy: 0.8928]
epoch: 82 Train [loss: 0.8622, accuracy: 0.8963], Valid [loss: 0.8668, accuracy: 0.8928]
epoch: 83 Train [loss: 0.8618, accuracy: 0.8966], Valid [loss: 0.8666, accuracy: 0.8936]
epoch: 84 Train [loss: 0.8616, accuracy: 0.8966], Valid [loss: 0.8665, accuracy: 0.8936]
epoch: 85 Train [loss: 0.8614, accuracy: 0.8966], Valid [loss: 0.8663, accuracy: 0.8936]
epoch: 86 Train [loss: 0.8611, accuracy: 0.8968], Valid [loss: 0.8661, accuracy: 0.8951]
epoch: 87 Train [loss: 0.8609, accuracy: 0.8970], Valid [loss: 0.8660, accuracy: 0.8943]
epoch: 88 Train [loss: 0.8606, accuracy: 0.8973], Valid [loss: 0.8659, accuracy: 0.8943]
epoch: 89 Train [loss: 0.8604, accuracy: 0.8976], Valid [loss: 0.8658, accuracy: 0.8958]
epoch: 90 Train [loss: 0.8603, accuracy: 0.8974], Valid [loss: 0.8656, accuracy: 0.8958]
epoch: 91 Train [loss: 0.8600, accuracy: 0.8978], Valid [loss: 0.8654, accuracy: 0.8951]
epoch: 92 Train [loss: 0.8597, accuracy: 0.8981], Valid [loss: 0.8652, accuracy: 0.8951]
epoch: 93 Train [loss: 0.8595, accuracy: 0.8981], Valid [loss: 0.8651, accuracy: 0.8951]
epoch: 94 Train [loss: 0.8593, accuracy: 0.8982], Valid [loss: 0.8650, accuracy: 0.8966]
epoch: 95 Train [loss: 0.8592, accuracy: 0.8979], Valid [loss: 0.8650, accuracy: 0.8966]
epoch: 96 Train [loss: 0.8589, accuracy: 0.8992], Valid [loss: 0.8648, accuracy: 0.8973]
epoch: 97 Train [loss: 0.8588, accuracy: 0.8990], Valid [loss: 0.8647, accuracy: 0.8973]
epoch: 98 Train [loss: 0.8586, accuracy: 0.8991], Valid [loss: 0.8645, accuracy: 0.8988]
epoch: 99 Train [loss: 0.8584, accuracy: 0.8992], Valid [loss: 0.8644, accuracy: 0.8981]
epoch: 100 Train [loss: 0.8581, accuracy: 0.9001], Valid [loss: 0.8643, accuracy: 0.8988]

Q.74
----------------
Train   [loss: 0.8567, accuracy: 0.9001]
Valid   [loss: 0.8574, accuracy: 0.8988]
Test    [loss: 0.8555, accuracy: 0.8966]
"""