import torch
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from scripts.study_case.ID_4.torch_geometric.nn import PointConv


def test_point_conv():
    in_channels, out_channels = (16, 32)
    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    num_nodes = edge_index.max().item() + 1
    x = torch.randn((num_nodes, in_channels))
    pos = torch.rand((num_nodes, 3))

    local_nn = Seq(Lin(in_channels + 3, 32), ReLU(), Lin(32, out_channels))
    global_nn = Seq(Lin(out_channels, out_channels))
    conv = PointConv(local_nn, global_nn)
    assert conv.__repr__() == (
        'PointConv(local_nn=Sequential(\n'
        '  (0): Linear(in_features=19, out_features=32, bias=True)\n'
        '  (1): ReLU()\n'
        '  (2): Linear(in_features=32, out_features=32, bias=True)\n'
        '), global_nn=Sequential(\n'
        '  (0): Linear(in_features=32, out_features=32, bias=True)\n'
        '))')
    assert conv(x, pos, edge_index).size() == (num_nodes, out_channels)
