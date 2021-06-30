from pytest import approx
import torch
from scripts.study_case.ID_13.torch_geometric.transforms import Polar
from scripts.study_case.ID_13.torch_geometric.data import Data


def test_polar():
    assert Polar().__repr__() == 'Polar(cat=True)'

    pos = torch.tensor([[-1, 0], [0, 0], [0, 2]], dtype=torch.float)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    data = Data(edge_index=edge_index, pos=pos)

    out = Polar()(data).edge_attr.view(-1).tolist()
    assert approx(out) == [0.5, 0, 0.5, 0.5, 1, 0.25, 1, 0.75]

    data.edge_attr = torch.tensor([1, 1, 1, 1], dtype=torch.float)
    out = Polar()(data).edge_attr.view(-1).tolist()
    assert approx(out) == [1, 0.5, 0, 1, 0.5, 0.5, 1, 1, 0.25, 1, 1, 0.75]
