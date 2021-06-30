import torch
from scripts.study_case.ID_4.torch_geometric.data import Batch
from scripts.study_case.ID_4.torch_geometric.nn import max_pool_x, max_pool


def test_max_pool_x():
    cluster = torch.tensor([0, 1, 0, 1, 2, 2])
    x = torch.Tensor([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
    batch = torch.tensor([0, 0, 0, 0, 1, 1])

    out = max_pool_x(cluster, x, batch)
    assert out[0].tolist() == [[5, 6], [7, 8], [11, 12]]
    assert out[1].tolist() == [0, 0, 1]

    out = max_pool_x(cluster, x, batch, size=2)
    assert out.tolist() == [[5, 6], [7, 8], [11, 12], [0, 0]]


def test_max_pool():
    cluster = torch.tensor([0, 1, 0, 1, 2, 2])
    x = torch.Tensor([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
    pos = torch.Tensor([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5]])
    edge_index = torch.tensor([[0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 5],
                               [1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2, 5, 4]])
    edge_attr = torch.Tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    batch = torch.tensor([0, 0, 0, 0, 1, 1])

    data = Batch(
        x=x, pos=pos, edge_index=edge_index, edge_attr=edge_attr, batch=batch)

    data = max_pool(cluster, data, transform=lambda x: x)

    assert data.x.tolist() == [[5, 6], [7, 8], [11, 12]]
    assert data.pos.tolist() == [[1, 1], [2, 2], [4.5, 4.5]]
    assert data.edge_index.tolist() == [[0, 1], [1, 0]]
    assert data.edge_attr.tolist() == [4, 4]
    assert data.batch.tolist() == [0, 0, 1]
