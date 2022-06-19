import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class NeuralNetwork(nn.Module):

    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

if __name__ == "__main__":

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} device'.format(device)) 
    # Using cuda device

    model = NeuralNetwork().to(device)
    print(model, end='\n\n')
    """
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

    X = torch.rand(1, 28, 28, device=device)
    logits = model(X)
    pred_probab = nn.Softmax(dim=1)(logits)
    y_pred = pred_probab.argmax(1)
    print(f'Predicted class: {y_pred}')
    # Predicted class: tensor([6], device='cuda:0')

    input_image = torch.rand(3, 28, 28)
    print(input_image.size())
    # torch.Size([3, 28, 28])

    # nn.Flatten 2次元（28x28）の画像を、1次元の784ピクセルの値へと変換
    flatten = nn.Flatten()
    flat_image = flatten(input_image)
    print(flat_image.size())
    # torch.Size([3, 784])

    # nn.Linear 線形変換
    layer1 = nn.Linear(in_features=28*28, out_features=20)
    hidden1 = layer1(flat_image)
    print(hidden1.size())
    # torch.Size([3, 20])

    # nn.ReLU
    print(f'Before ReLU: {hidden1}\n\n')
    """
    Before ReLU: tensor([[ 0.3494, -0.0028, -0.1955, -0.2106,  0.3906, -0.0010,  0.1782, -0.0624,
        0.0560,  0.5409,  0.1923, -0.0664,  0.6123, -0.1223, -0.8349,  0.3310,
        0.1597,  0.3237, -0.0100,  0.1365],
        [ 0.4408, -0.0262, -0.3927, -0.0991,  0.3410, -0.1739,  0.0389, -0.1528,
        -0.1290,  0.1481,  0.4190,  0.2112,  0.3446,  0.1171, -0.8175,  0.4172,
        -0.0545,  0.1920, -0.2307, -0.2221],
        [ 0.2272, -0.2197, -0.2614, -0.1443,  0.3050, -0.1097, -0.0115,  0.1251,
        -0.0047,  0.3379,  0.2928,  0.2923,  0.2579,  0.1119, -0.9612,  0.0020,
        0.3277,  0.2285, -0.0396, -0.0673]], grad_fn=<AddmmBackward0>)
    """
    hidden1 = nn.ReLU()(hidden1)
    print(f'After ReLU: {hidden1}')
    """
    After ReLU: tensor([[0.3494, 0.0000, 0.0000, 0.0000, 0.3906, 0.0000, 0.1782, 0.0000, 0.0560,
         0.5409, 0.1923, 0.0000, 0.6123, 0.0000, 0.0000, 0.3310, 0.1597, 0.3237,
         0.0000, 0.1365],
        [0.4408, 0.0000, 0.0000, 0.0000, 0.3410, 0.0000, 0.0389, 0.0000, 0.0000,
         0.1481, 0.4190, 0.2112, 0.3446, 0.1171, 0.0000, 0.4172, 0.0000, 0.1920,
         0.0000, 0.0000],
        [0.2272, 0.0000, 0.0000, 0.0000, 0.3050, 0.0000, 0.0000, 0.1251, 0.0000,
         0.3379, 0.2928, 0.2923, 0.2579, 0.1119, 0.0000, 0.0020, 0.3277, 0.2285,
         0.0000, 0.0000]], grad_fn=<ReluBackward0>)
    """

    # nn.Sequential
    seq_modules = nn.Sequential(
        flatten,
        layer1,
        nn.ReLU(),
        nn.Linear(20, 10)
    )
    input_image = torch.rand(3, 28, 28)
    logits = seq_modules(input_image)
    print(logits)
    """
    tensor([[ 0.3350, -0.2372,  0.0675,  0.3730,  0.1177,  0.2763,  0.1675,  0.0841,
          0.0890, -0.0049],
        [ 0.2068, -0.2246,  0.0953,  0.3266,  0.0973,  0.1857,  0.1851, -0.0117,
         -0.0204, -0.1994],
        [ 0.1726, -0.2419,  0.0975,  0.2035,  0.0627,  0.1710,  0.2193, -0.0372,
         -0.0248, -0.0491]], grad_fn=<AddmmBackward0>)
    """

    # nn.Softmax
    softmax = nn.Softmax(dim=1)
    pred_probab = softmax(logits)
    print(pred_probab)
    """
    tensor([[0.1215, 0.0685, 0.0929, 0.1262, 0.0977, 0.1145, 0.1027, 0.0945, 0.0950,
         0.0865],
        [0.1137, 0.0739, 0.1017, 0.1282, 0.1019, 0.1114, 0.1113, 0.0914, 0.0906,
         0.0758],
        [0.1112, 0.0735, 0.1031, 0.1147, 0.0996, 0.1110, 0.1165, 0.0901, 0.0913,
         0.0891]], grad_fn=<SoftmaxBackward0>)
    """

    print('Model structure: ', model, '\n\n')
    """
    Model structure:  NeuralNetwork(
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
    for name, param in model.named_parameters():
        print(f'Layer: {name} | Size: {param.size()} | Values: {param[:2]} \n')
    """
    Layer: linear_relu_stack.0.weight | Size: torch.Size([512, 784]) | Values: tensor([[ 0.0036,  0.0109,  0.0092,  ..., -0.0209, -0.0172,  0.0270],
        [ 0.0243, -0.0285,  0.0161,  ...,  0.0217, -0.0351,  0.0292]],
       device='cuda:0', grad_fn=<SliceBackward0>)

    Layer: linear_relu_stack.0.bias | Size: torch.Size([512]) | Values: tensor([0.0246, 0.0309], device='cuda:0', grad_fn=<SliceBackward0>)

    Layer: linear_relu_stack.2.weight | Size: torch.Size([512, 512]) | Values: tensor([[ 0.0264,  0.0234,  0.0291,  ..., -0.0082,  0.0219, -0.0027],
            [-0.0243,  0.0054,  0.0181,  ...,  0.0373,  0.0129,  0.0138]],
        device='cuda:0', grad_fn=<SliceBackward0>)

    Layer: linear_relu_stack.2.bias | Size: torch.Size([512]) | Values: tensor([-0.0217,  0.0415], device='cuda:0', grad_fn=<SliceBackward0>)

    Layer: linear_relu_stack.4.weight | Size: torch.Size([10, 512]) | Values: tensor([[-0.0376,  0.0298, -0.0043,  ...,  0.0352,  0.0074, -0.0423],
            [ 0.0432,  0.0216,  0.0437,  ..., -0.0364,  0.0018,  0.0436]],
        device='cuda:0', grad_fn=<SliceBackward0>)

    Layer: linear_relu_stack.4.bias | Size: torch.Size([10]) | Values: tensor([0.0374, 0.0143], device='cuda:0', grad_fn=<SliceBackward0>)
    """