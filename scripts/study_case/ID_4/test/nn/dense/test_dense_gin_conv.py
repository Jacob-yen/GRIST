import torch
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from scripts.study_case.ID_4.torch_geometric.nn import GINConv, DenseGINConv


def test_dense_sage_conv():
    channels = 16
    nn = Seq(Lin(channels, channels), ReLU(), Lin(channels, channels))
    sparse_conv = GINConv(nn)
    dense_conv = DenseGINConv(nn)
    dense_conv = DenseGINConv(nn, train_eps=True)
    assert dense_conv.__repr__() == (
        'DenseGINConv(nn=Sequential(\n'
        '  (0): Linear(in_features=16, out_features=16, bias=True)\n'
        '  (1): ReLU()\n'
        '  (2): Linear(in_features=16, out_features=16, bias=True)\n'
        '))')

    x = torch.randn((5, channels))
    edge_index = torch.tensor([[0, 0, 1, 1, 2, 2, 3, 4],
                               [1, 2, 0, 2, 0, 1, 4, 3]])

    sparse_out = sparse_conv(x, edge_index)
    assert sparse_out.size() == (5, channels)

    x = torch.cat([x, x.new_zeros(1, channels)], dim=0).view(2, 3, channels)
    adj = torch.Tensor([
        [
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0],
        ],
        [
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, 0],
        ],
    ])
    mask = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.uint8)

    dense_out = dense_conv(x, adj, mask)
    assert dense_out.size() == (2, 3, channels)

    assert dense_out[1, 2].abs().sum().item() == 0
    dense_out = dense_out.view(6, channels)[:-1]
    assert torch.allclose(sparse_out, dense_out, atol=1e-04)


def test_dense_gin_conv_with_broadcasting():
    batch_size, num_nodes, channels = 8, 3, 16
    nn = Seq(Lin(channels, channels), ReLU(), Lin(channels, channels))
    conv = DenseGINConv(nn)

    x = torch.randn(batch_size, num_nodes, channels)
    adj = torch.Tensor([
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0],
    ])

    assert conv(x, adj).size() == (batch_size, num_nodes, channels)
    mask = torch.tensor([1, 1, 1], dtype=torch.uint8)
    assert conv(x, adj, mask).size() == (batch_size, num_nodes, channels)
