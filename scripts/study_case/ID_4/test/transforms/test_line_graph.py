import torch
from scripts.study_case.ID_4.torch_geometric.transforms import LineGraph
from scripts.study_case.ID_4.torch_geometric.data import Data


def test_line_graph():
    assert LineGraph().__repr__() == 'LineGraph()'

    # Directed.
    edge_index = torch.tensor([
        [0, 1, 2, 2, 3],
        [1, 2, 0, 3, 0],
    ])
    data = Data(edge_index=edge_index)
    data = LineGraph()(data)
    assert data.edge_index.tolist() == [[0, 1, 1, 2, 3, 4], [1, 2, 3, 0, 4, 0]]

    # Undirected.
    edge_index = torch.tensor([[0, 0, 0, 1, 1, 2, 2, 3, 3, 3, 4, 4],
                               [1, 2, 3, 0, 4, 0, 3, 0, 2, 4, 1, 3]])
    edge_attr = torch.ones(edge_index.size(1))
    data = Data(edge_index=edge_index, edge_attr=edge_attr)
    data = LineGraph()(data)
    assert data.edge_index.max().item() + 1 == data.x.size(0)
    assert data.edge_index.tolist() == [
        [0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 4, 4, 4, 5, 5, 5],
        [1, 2, 3, 0, 2, 4, 0, 1, 4, 5, 0, 5, 1, 2, 5, 2, 3, 4],
    ]
    assert data.x.tolist() == [2, 2, 2, 2, 2, 2]
