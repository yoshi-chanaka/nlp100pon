import torch

if __name__ == "__main__":

    x = torch.ones(5)
    y = torch.zeros(3)
    w = torch.randn(5, 3, requires_grad=True)
    b = torch.randn(3, requires_grad=True)
    z = torch.matmul(x, w) + b
    loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)

    print('Gradient function for z =', z.grad_fn)
    print('Gradient function for loss =', loss.grad_fn)
    """
    Gradient function for z = <AddBackward0 object at 0x7f9b6c67d080>
    Gradient function for loss = <BinaryCrossEntropyWithLogitsBackward0 object at 0x7f9b6c67d080>
    """

    loss.backward()
    print(w.grad)
    print(b.grad)
    """
    tensor([[0.3069, 0.3169, 0.2791],
            [0.3069, 0.3169, 0.2791],
            [0.3069, 0.3169, 0.2791],
            [0.3069, 0.3169, 0.2791],
            [0.3069, 0.3169, 0.2791]])
    tensor([0.3069, 0.3169, 0.2791])
    """

    # 勾配計算をしない
    # no_grad()のパターン
    z = torch.matmul(x, w) + b
    print(z.requires_grad)
    # True

    with torch.no_grad():
        z = torch.matmul(x, w) + b
    print(z.requires_grad)
    # False

    # detach()のパターン
    z = torch.matmul(x, w) + b
    z_det = z.detach()
    print(z_det.requires_grad)
    # False

    inp = torch.eye(5, requires_grad=True)
    out = (inp + 1).pow(2)
    out.backward(torch.ones_like(inp), retain_graph=True)
    print('First call\n', inp.grad)
    """
    First call
    tensor([[4., 2., 2., 2., 2.],
            [2., 4., 2., 2., 2.],
            [2., 2., 4., 2., 2.],
            [2., 2., 2., 4., 2.],
            [2., 2., 2., 2., 4.]])
    """
    out.backward(torch.ones_like(inp), retain_graph=True)
    print('\nSecond call\n', inp.grad)
    """
    Second call
    tensor([[8., 4., 4., 4., 4.],
            [4., 8., 4., 4., 4.],
            [4., 4., 8., 4., 4.],
            [4., 4., 4., 8., 4.],
            [4., 4., 4., 4., 8.]])
    """
    # 実際にPyTorchでディープラーニングモデルの訓練を行う際には、オプティマイザー（optimizer）が、勾配をリセットする役割を担ってくれます
    inp.grad.zero_()
    out.backward(torch.ones_like(inp), retain_graph=True)
    print('\nCall after zeroing gradients\n', inp.grad)
    """
    Call after zeroing gradients
    tensor([[4., 2., 2., 2., 2.],
            [2., 4., 2., 2., 2.],
            [2., 2., 4., 2., 2.],
            [2., 2., 2., 4., 2.],
            [2., 2., 2., 2., 4.]])
    """
