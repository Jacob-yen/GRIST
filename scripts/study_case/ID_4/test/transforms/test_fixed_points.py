import torch
from scripts.study_case.ID_4.torch_geometric.transforms import FixedPoints
from scripts.study_case.ID_4.torch_geometric.data import Data


def test_fixed_points():
    assert FixedPoints(1024).__repr__() == 'FixedPoints(1024)'

    data = Data(
        pos=torch.randn(100, 3), x=torch.randn(100, 16), y=torch.randn(1),
        edge_attr=torch.randn(100, 3))
    data = FixedPoints(50)(data)
    assert len(data) == 4
    assert data.pos.size() == (50, 3)
    assert data.x.size() == (50, 16)
    assert data.y.size() == (1, )
    assert data.edge_attr.size() == (100, 3)

    data = Data(
        pos=torch.randn(100, 3), x=torch.randn(100, 16), y=torch.randn(1),
        edge_attr=torch.randn(100, 3))
    data = FixedPoints(200)(data)
    assert len(data) == 4
    assert data.pos.size() == (200, 3)
    assert data.x.size() == (200, 16)
    assert data.y.size() == (1, )
    assert data.edge_attr.size() == (100, 3)
