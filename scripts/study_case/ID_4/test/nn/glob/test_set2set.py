import torch
from scripts.study_case.ID_4.torch_geometric.nn import Set2Set


def test_set2set():
    set2set = Set2Set(in_channels=2, processing_steps=1)
    assert set2set.__repr__() == 'Set2Set(2, 4)'

    N = 4
    x_1, batch_1 = torch.randn(N, 2), torch.zeros(N, dtype=torch.long)
    out_1 = set2set(x_1, batch_1).view(-1)

    N = 6
    x_2, batch_2 = torch.randn(N, 2), torch.zeros(N, dtype=torch.long)
    out_2 = set2set(x_2, batch_2).view(-1)

    x, batch = torch.cat([x_1, x_2]), torch.cat([batch_1, batch_2 + 1])
    out = set2set(x, batch)
    assert out.size() == (2, 4)
    assert out_1.tolist() == out[0].tolist()
    assert out_2.tolist() == out[1].tolist()

    x, batch = torch.cat([x_2, x_1]), torch.cat([batch_2, batch_1 + 1])
    out = set2set(x, batch)
    assert out.size() == (2, 4)
    assert out_1.tolist() == out[1].tolist()
    assert out_2.tolist() == out[0].tolist()
