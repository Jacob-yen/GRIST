import torch
from scripts.study_case.ID_4.torch_geometric.nn import graclus


def test_graclus():
    edge_index = torch.tensor([[0, 1], [1, 0]])
    assert graclus(edge_index).tolist() == [0, 0]
