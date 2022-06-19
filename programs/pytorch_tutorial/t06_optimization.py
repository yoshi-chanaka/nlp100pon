import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
from t04_buildmodel import NeuralNetwork


def train_loop(dataloader, model, loss_fn, optimizer, device):
    
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # 予測と損失の計算
        X, y = X.to(device), y.to(device) # 
        pred = model(X)
        loss = loss_fn(pred, y)

        # バックプロパゲーション
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss.cpu() # 
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f'loss: {loss:>7f} [{current:>5d}/{size:>5d}]')

def test_loop(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device) # 
            pred = model(X)
            test_loss += loss_fn(pred, y).cpu().item() # 
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        test_loss /= size
        correct /= size
        print(f'Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n')


if __name__ == "__main__":

    training_data = datasets.FashionMNIST(
        root='data',
        train=True,
        download=True,
        transform=ToTensor()
    )

    test_data = datasets.FashionMNIST(
        root='data',
        train=False,
        download=True,
        transform=ToTensor()
    )

    train_dataloader = DataLoader(training_data, batch_size=64)
    test_dataloader = DataLoader(test_data, batch_size=64)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = NeuralNetwork()
    model.to(device)

    # ハイパーパラメータ
    learning_rate = 1e-3
    batch_size = 64

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

    epochs = 10
    for t in range(epochs):
        print(f'Epoch {t + 1}\n-------------------------------')
        train_loop(train_dataloader, model, loss_fn, optimizer, device)
        test_loop(test_dataloader, model, loss_fn, device)
    print('Done!')

"""
Epoch 1
-------------------------------
loss: 2.304941 [    0/60000]
loss: 2.292183 [ 6400/60000]
loss: 2.279816 [12800/60000]
loss: 2.280916 [19200/60000]
loss: 2.269215 [25600/60000]
loss: 2.237870 [32000/60000]
loss: 2.250041 [38400/60000]
loss: 2.223077 [44800/60000]
loss: 2.222422 [51200/60000]
loss: 2.188133 [57600/60000]
Test Error:
 Accuracy: 38.4%, Avg loss: 0.034491

Epoch 2
-------------------------------
loss: 2.197459 [    0/60000]
loss: 2.193916 [ 6400/60000]
loss: 2.148626 [12800/60000]
loss: 2.170220 [19200/60000]
loss: 2.133434 [25600/60000]
loss: 2.071163 [32000/60000]
loss: 2.105829 [38400/60000]
loss: 2.041654 [44800/60000]
loss: 2.059407 [51200/60000]
loss: 1.986040 [57600/60000]
Test Error:
 Accuracy: 52.4%, Avg loss: 0.031436

Epoch 3
-------------------------------
loss: 2.020787 [    0/60000]
loss: 2.005905 [ 6400/60000]
loss: 1.907688 [12800/60000]
loss: 1.955570 [19200/60000]
loss: 1.881438 [25600/60000]
loss: 1.789744 [32000/60000]
loss: 1.843488 [38400/60000]
loss: 1.733624 [44800/60000]
loss: 1.793281 [51200/60000]
loss: 1.669010 [57600/60000]
Test Error:
 Accuracy: 61.4%, Avg loss: 0.026651

Epoch 4
-------------------------------
loss: 1.751198 [    0/60000]
loss: 1.724298 [ 6400/60000]
loss: 1.577246 [12800/60000]
loss: 1.649532 [19200/60000]
loss: 1.597481 [25600/60000]
loss: 1.485443 [32000/60000]
loss: 1.545816 [38400/60000]
loss: 1.428240 [44800/60000]
loss: 1.542688 [51200/60000]
loss: 1.373208 [57600/60000]
Test Error:
 Accuracy: 63.6%, Avg loss: 0.022429

Epoch 5
-------------------------------
loss: 1.520230 [    0/60000]
loss: 1.493661 [ 6400/60000]
loss: 1.323194 [12800/60000]
loss: 1.412153 [19200/60000]
loss: 1.415524 [25600/60000]
loss: 1.277206 [32000/60000]
loss: 1.347286 [38400/60000]
loss: 1.233909 [44800/60000]
loss: 1.388965 [51200/60000]
loss: 1.193557 [57600/60000]
Test Error:
 Accuracy: 64.4%, Avg loss: 0.019878

Epoch 6
-------------------------------
loss: 1.373219 [    0/60000]
loss: 1.354802 [ 6400/60000]
loss: 1.164783 [12800/60000]
loss: 1.271827 [19200/60000]
loss: 1.308631 [25600/60000]
loss: 1.146207 [32000/60000]
loss: 1.232084 [38400/60000]
loss: 1.118341 [44800/60000]
loss: 1.292686 [51200/60000]
loss: 1.087856 [57600/60000]
Test Error:
 Accuracy: 65.0%, Avg loss: 0.018309

Epoch 7
-------------------------------
loss: 1.274415 [    0/60000]
loss: 1.270011 [ 6400/60000]
loss: 1.059883 [12800/60000]
loss: 1.181626 [19200/60000]
loss: 1.237032 [25600/60000]
loss: 1.057079 [32000/60000]
loss: 1.158959 [38400/60000]
loss: 1.046053 [44800/60000]
loss: 1.224539 [51200/60000]
loss: 1.019114 [57600/60000]
Test Error:
 Accuracy: 65.5%, Avg loss: 0.017241

Epoch 8
-------------------------------
loss: 1.200456 [    0/60000]
loss: 1.212828 [ 6400/60000]
loss: 0.984133 [12800/60000]
loss: 1.117595 [19200/60000]
loss: 1.183769 [25600/60000]
loss: 0.992252 [32000/60000]
loss: 1.107563 [38400/60000]
loss: 0.998182 [44800/60000]
loss: 1.173818 [51200/60000]
loss: 0.969760 [57600/60000]
Test Error:
 Accuracy: 65.7%, Avg loss: 0.016464

Epoch 9
-------------------------------
loss: 1.142365 [    0/60000]
loss: 1.170468 [ 6400/60000]
loss: 0.926739 [12800/60000]
loss: 1.069668 [19200/60000]
loss: 1.142069 [25600/60000]
loss: 0.943390 [32000/60000]
loss: 1.069593 [38400/60000]
loss: 0.965762 [44800/60000]
loss: 1.135751 [51200/60000]
loss: 0.931570 [57600/60000]
Test Error:
 Accuracy: 66.3%, Avg loss: 0.015874

Epoch 10
-------------------------------
loss: 1.095962 [    0/60000]
loss: 1.136269 [ 6400/60000]
loss: 0.881973 [12800/60000]
loss: 1.032414 [19200/60000]
loss: 1.108507 [25600/60000]
loss: 0.905621 [32000/60000]
loss: 1.038499 [38400/60000]
loss: 0.943130 [44800/60000]
loss: 1.106601 [51200/60000]
loss: 0.900940 [57600/60000]
Test Error:
 Accuracy: 66.8%, Avg loss: 0.015410

Done!
"""

