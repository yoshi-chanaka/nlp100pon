import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
from t04_buildmodel import NeuralNetwork

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y) # cross entropy

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f'loss: {loss:>7f} [{current:>5d}/{size:>5d}]')


def test(dataloader, model):
    size = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss = loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= size
    correct /= size
    print(f'Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n')



if __name__ == "__main__":

    training_data = datasets.FashionMNIST(
        root='data',
        train=True,
        download=True,
        transform=ToTensor(),
    )

    test_data = datasets.FashionMNIST(
        root='data',
        train=False,
        download=True,
        transform=ToTensor(),
    )

    batch_size = 64

    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    for X, y in test_dataloader:
        print('Shape of X [N, C, H, W]: ', X.shape)
        print('Shape of y: ', y.shape, y.dtype)
        break

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} device'.format(device))

    model = NeuralNetwork().to(device)
    print(model)

    """
    Shape of X [N, C, H, W]:  torch.Size([64, 1, 28, 28])
    Shape of y:  torch.Size([64]) torch.int64
    Using cuda device
    NeuralNetwork(
        (flatten): Flatten(start_dim=1, end_dim=-1)
        (linear_relu_stack): Sequential(
            (0): Linear(in_features=784, out_features=512, bias=True)
            (1): ReLU()
            (2): Linear(in_features=512, out_features=512, bias=True)
            (3): ReLU()
            (4): Linear(in_features=512, out_features=10, bias=True)
            (5): ReLU()
        )
    )
    """

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    epochs = 5
    for t in range(epochs):
        print(f'Epoch {t + 1}\n-------------------------------')
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model)

    print('Done!')

    """
    Epoch 1
    -------------------------------
    loss: 2.296344 [    0/60000]
    loss: 2.290899 [ 6400/60000]
    loss: 2.285473 [12800/60000]
    loss: 2.292129 [19200/60000]
    loss: 2.282927 [25600/60000]
    loss: 2.263345 [32000/60000]
    loss: 2.257121 [38400/60000]
    loss: 2.240538 [44800/60000]
    loss: 2.224277 [51200/60000]
    loss: 2.247985 [57600/60000]
    Test Error:
    Accuracy: 39.1%, Avg loss: 0.000228

    Epoch 2
    -------------------------------
    loss: 2.209234 [    0/60000]
    loss: 2.228206 [ 6400/60000]
    loss: 2.215657 [12800/60000]
    loss: 2.244287 [19200/60000]
    loss: 2.222862 [25600/60000]
    loss: 2.193680 [32000/60000]
    loss: 2.171337 [38400/60000]
    loss: 2.149064 [44800/60000]
    loss: 2.120661 [51200/60000]
    loss: 2.172623 [57600/60000]
    Test Error:
    Accuracy: 41.4%, Avg loss: 0.000223

    Epoch 3
    -------------------------------
    loss: 2.102762 [    0/60000]
    loss: 2.132894 [ 6400/60000]
    loss: 2.116169 [12800/60000]
    loss: 2.165618 [19200/60000]
    loss: 2.139341 [25600/60000]
    loss: 2.100152 [32000/60000]
    loss: 2.052669 [38400/60000]
    loss: 2.031531 [44800/60000]
    loss: 1.985861 [51200/60000]
    loss: 2.073052 [57600/60000]
    Test Error:
    Accuracy: 43.8%, Avg loss: 0.000218

    Epoch 4
    -------------------------------
    loss: 1.969479 [    0/60000]
    loss: 2.013812 [ 6400/60000]
    loss: 1.996822 [12800/60000]
    loss: 2.066962 [19200/60000]
    loss: 2.048469 [25600/60000]
    loss: 2.000619 [32000/60000]
    loss: 1.923213 [38400/60000]
    loss: 1.915552 [44800/60000]
    loss: 1.856944 [51200/60000]
    loss: 1.973550 [57600/60000]
    Test Error:
    Accuracy: 44.9%, Avg loss: 0.000212

    Epoch 5
    -------------------------------
    loss: 1.844776 [    0/60000]
    loss: 1.901480 [ 6400/60000]
    loss: 1.887967 [12800/60000]
    loss: 1.975253 [19200/60000]
    loss: 1.964870 [25600/60000]
    loss: 1.915282 [32000/60000]
    loss: 1.817122 [38400/60000]
    loss: 1.826598 [44800/60000]
    loss: 1.757222 [51200/60000]
    loss: 1.892194 [57600/60000]
    Test Error:
    Accuracy: 46.2%, Avg loss: 0.000206

    Done!
    """


