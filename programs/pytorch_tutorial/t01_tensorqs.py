import torch
import numpy as np

if __name__ == "__main__":

    data = [[1, 2], [3, 4]]
    x_data = torch.tensor(data)
    print(x_data)
    """
    tensor([[1, 2],
            [3, 4]])
    """

    np_array = np.array(data)
    x_np = torch.from_numpy (np_array)
    print(x_np)
    """
    tensor([[1, 2],
            [3, 4]])
    """

    x_ones = torch.ones_like(x_data)
    print(f'Ones Tensor: \n {x_ones} \n')

    x_rand = torch.rand_like(x_data, dtype=torch.float)
    print(f'Random Tensor: \n {x_rand} \n')
    """
    Ones Tensor:
    tensor([[1, 1],
            [1, 1]])

    Random Tensor:
    tensor([[0.7700, 0.1026],
            [0.9910, 0.0481]])
    """

    # ランダム値や定数のテンソルの作成
    shape = (2, 3,)
    rand_tensor = torch.rand(shape)
    ones_tensor = torch.ones(shape)
    zeros_tensor = torch.zeros(shape)
    print(f'Random Tensor: \n {rand_tensor} \n')
    print(f'Ones Tensor: \n {ones_tensor} \n')
    print(f'Zeros Tensor: \n {zeros_tensor} \n')
    """
    Random Tensor:
    tensor([[0.1760, 0.2447, 0.2228],
            [0.7992, 0.3418, 0.1608]])

    Ones Tensor:
    tensor([[1., 1., 1.],
            [1., 1., 1.]])

    Zeros Tensor:
    tensor([[0., 0., 0.],
            [0., 0., 0.]])
    """

    tensor = torch.rand(3, 4)
    print(f'Shape of tensor: {tensor.shape}')
    print(f'Datatype of tensor: {tensor.dtype}')
    print(f'Device tensor is stored on: {tensor.device}')

    """
    Shape of tensor: torch.Size([3, 4])
    Datatype of tensor: torch.float32
    Device tensor is stored on: cpu
    """

    if torch.cuda.is_available():
        tensor = tensor.to('cuda')
        print(f'Device tensor is stored on: {tensor.device}')
        # Device tensor is stored on: cuda:0
    tensor = tensor.to('cpu')
    print(f'Device tensor is stored on: {tensor.device}')
    # Device tensor is stored on: cpu

    tensor = torch.ones(4, 4)
    print('First row: ', tensor[0])
    print('First column: ', tensor[:, 0])
    print('Last column: ', tensor[..., -1])

    tensor[:, 1] = 0
    print(tensor)
    """
    First row:  tensor([1., 1., 1., 1.])
    First column:  tensor([1., 1., 1., 1.])
    Last column:  tensor([1., 1., 1., 1.])
    tensor([[1., 0., 1., 1.],
            [1., 0., 1., 1.],
            [1., 0., 1., 1.],
            [1., 0., 1., 1.]])
    """

    t1 = torch.cat([tensor, tensor, tensor], dim=1)
    print(t1)
    """
    tensor([[1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
        [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
        [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
        [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.]])
    """

    # 算術演算
    y1 = tensor @ tensor.T
    y2 = tensor.matmul(tensor.T)
    y3 = torch.rand_like(tensor)
    torch.matmul(tensor, tensor.T, out=y3)
    print(f'y1: {y1}\ny2: {y2}\ny3: {y3}\n')

    z1 = tensor * tensor
    z2 = tensor.mul(tensor)
    z3 = torch.rand_like(tensor)
    torch.mul(tensor, tensor, out=z3)
    print(f'z1: {z1}\nz2: {z2}\nz3: {z3}\n')
    """
    y1: tensor([[3., 3., 3., 3.],
        [3., 3., 3., 3.],
        [3., 3., 3., 3.],
        [3., 3., 3., 3.]])
    y2: tensor([[3., 3., 3., 3.],
            [3., 3., 3., 3.],
            [3., 3., 3., 3.],
            [3., 3., 3., 3.]])
    y3: tensor([[3., 3., 3., 3.],
            [3., 3., 3., 3.],
            [3., 3., 3., 3.],
            [3., 3., 3., 3.]])

    z1: tensor([[1., 0., 1., 1.],
            [1., 0., 1., 1.],
            [1., 0., 1., 1.],
            [1., 0., 1., 1.]])
    z2: tensor([[1., 0., 1., 1.],
            [1., 0., 1., 1.],
            [1., 0., 1., 1.],
            [1., 0., 1., 1.]])
    z3: tensor([[1., 0., 1., 1.],
            [1., 0., 1., 1.],
            [1., 0., 1., 1.],
            [1., 0., 1., 1.]])
    """

    # 1要素のテンソル
    agg = tensor.sum()
    agg_item = agg.item()
    print(agg_item, type(agg_item))
    # 12.0 <class 'float'>

    # インプレース操作
    print(tensor, '\n')
    tensor.add_(5)
    print(tensor)
    """
    tensor([[1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.]])

    tensor([[6., 5., 6., 6.],
        [6., 5., 6., 6.],
        [6., 5., 6., 6.],
        [6., 5., 6., 6.]])
    """

    # Tensor to Numpy array
    t = torch.ones(5)
    print(f't: {t}')
    n = t.numpy()
    print(f'n: {n}')
    """
    t: tensor([1., 1., 1., 1., 1.])
    n: [1. 1. 1. 1. 1.]
    """

    t.add_(1)
    print(f't: {t}')
    print(f'n: {n}')
    """
    t: tensor([2., 2., 2., 2., 2.])
    n: [2. 2. 2. 2. 2.]
    """
    
    # Numpy array to Tensor
    n = np.ones(5)
    t = torch.from_numpy(n)

    np.add(n, 1, out=n)
    print(f't: {t}')
    print(f'n: {n}')
    """
    t: tensor([2., 2., 2., 2., 2.], dtype=torch.float64)
    n: [2. 2. 2. 2. 2.]
    """




