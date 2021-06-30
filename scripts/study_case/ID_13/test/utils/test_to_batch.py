import torch
from scripts.study_case.ID_13.torch_geometric.utils import to_batch


def test_to_batch():
    x = torch.Tensor([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
    batch = torch.tensor([0, 0, 1, 2, 2, 2])

    x, num_nodes = to_batch(x, batch)
    expected = [
        [[1, 2], [3, 4], [0, 0]],
        [[5, 6], [0, 0], [0, 0]],
        [[7, 8], [9, 10], [11, 12]],
    ]
    assert x.tolist() == expected
    assert num_nodes.tolist() == [2, 1, 3]
