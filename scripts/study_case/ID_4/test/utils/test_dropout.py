import torch
from scripts.study_case.ID_4.torch_geometric.utils import dropout_adj


def test_dropout_adj():
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])
    edge_attr = torch.Tensor([1, 2, 3, 4, 5, 6])

    out = dropout_adj(edge_index, edge_attr, training=False)
    assert edge_index.tolist() == out[0].tolist()
    assert edge_attr.tolist() == out[1].tolist()

    torch.manual_seed(5)
    out = dropout_adj(edge_index, edge_attr)
    assert out[0].tolist() == [[1, 3], [0, 2]]
    assert out[1].tolist() == [2, 6]

    torch.manual_seed(5)
    out = dropout_adj(edge_index, edge_attr, force_undirected=True)
    assert out[0].tolist() == [[1, 2], [2, 1]]
    assert out[1].tolist() == [3, 3]
