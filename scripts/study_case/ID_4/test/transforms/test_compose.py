import torch
import scripts.study_case.ID_4.torch_geometric.transforms as T
from scripts.study_case.ID_4.torch_geometric.data import Data


def test_compose():
    transform = T.Compose([T.Center(), T.AddSelfLoops()])
    assert transform.__repr__() == ('Compose([\n'
                                    '    Center(),\n'
                                    '    AddSelfLoops(),\n'
                                    '])')

    pos = torch.Tensor([[0, 0], [2, 0], [4, 0]])
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])

    data = Data(edge_index=edge_index, pos=pos)
    data = transform(data)
    assert len(data) == 2
    assert data.pos.tolist() == [[-2, 0], [0, 0], [2, 0]]
    assert data.edge_index.tolist() == [[0, 0, 1, 1, 1, 2, 2],
                                        [0, 1, 0, 1, 2, 1, 2]]
