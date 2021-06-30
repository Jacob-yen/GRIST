import torch
from scripts.study_case.ID_13.torch_geometric.transforms import LocalCartesian
from scripts.study_case.ID_13.torch_geometric.data import Data


def test_local_cartesian():
    assert LocalCartesian().__repr__() == 'LocalCartesian(cat=True)'

    pos = torch.tensor([[-1, 0], [0, 0], [2, 0]], dtype=torch.float)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    data = Data(edge_index=edge_index, pos=pos)

    out = LocalCartesian()(data).edge_attr.tolist()
    assert out == [[1, 0.5], [0.25, 0.5], [1, 0.5], [0, 0.5]]

    data.edge_attr = torch.tensor([1, 1, 1, 1], dtype=torch.float)
    out = LocalCartesian()(data).edge_attr.tolist()
    assert out == [[1, 1, 0.5], [1, 0.25, 0.5], [1, 1, 0.5], [1, 0, 0.5]]
