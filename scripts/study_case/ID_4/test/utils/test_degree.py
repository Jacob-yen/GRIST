import torch
from scripts.study_case.ID_4.torch_geometric.utils import degree


def test_degree():
    row = torch.tensor([0, 1, 0, 2, 0])
    deg = degree(row, dtype=torch.long)
    assert deg.dtype == torch.long
    assert deg.tolist() == [3, 1, 1]
