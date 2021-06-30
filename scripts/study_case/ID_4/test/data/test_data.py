import torch
import scripts.study_case.ID_4.torch_geometric
from scripts.study_case.ID_4.torch_geometric.data import Data
from torch_sparse import coalesce


def test_data():
    torch_geometric.set_debug(True)

    x = torch.tensor([[1, 3, 5], [2, 4, 6]], dtype=torch.float).t()
    edge_index = torch.tensor([[0, 0, 1, 1, 2], [1, 1, 0, 2, 1]])
    data = Data(x=x, edge_index=edge_index).to(torch.device('cpu'))

    N = data.num_nodes

    assert data.x.tolist() == x.tolist()
    assert data['x'].tolist() == x.tolist()

    assert sorted(data.keys) == ['edge_index', 'x']
    assert len(data) == 2
    assert 'x' in data and 'edge_index' in data and 'pos' not in data

    assert data.__cat_dim__('x', data.x) == 0
    assert data.__cat_dim__('edge_index', data.edge_index) == -1
    assert data.__inc__('x', data.x) == 0
    assert data.__inc__('edge_index', data.edge_index) == data.num_nodes

    assert not data.x.is_contiguous()
    data.contiguous()
    assert data.x.is_contiguous()

    assert not data.is_coalesced()
    data.edge_index, _ = coalesce(data.edge_index, None, N, N)
    data = data.coalesce()
    assert data.is_coalesced()

    clone = data.clone()
    assert clone != data
    assert len(clone) == len(data)
    assert clone.x.tolist() == data.x.tolist()
    assert clone.edge_index.tolist() == data.edge_index.tolist()

    data['x'] = x + 1
    assert data.x.tolist() == (x + 1).tolist()

    assert data.__repr__() == 'Data(edge_index=[2, 4], x=[3, 2])'

    dictionary = {'x': data.x, 'edge_index': data.edge_index}
    data = Data.from_dict(dictionary)
    assert sorted(data.keys) == ['edge_index', 'x']

    assert not data.contains_isolated_nodes()
    assert not data.contains_self_loops()
    assert data.is_undirected()
    assert not data.is_directed()

    assert data.num_nodes == 3
    assert data.num_edges == 4
    assert data.num_faces is None
    assert data.num_node_features == 2
    assert data.num_features == 2

    data.edge_attr = torch.randn(data.num_edges, 2)
    assert data.num_edge_features == 2
    data.edge_attr = None

    data.x = None
    assert data.num_nodes == 3

    data.edge_index = None
    assert data.num_nodes is None
    assert data.num_edges is None

    data.num_nodes = 4
    assert data.num_nodes == 4

    data = Data(x=x, attribute=x)
    assert len(data) == 2
    assert data.x.tolist() == x.tolist()
    assert data.attribute.tolist() == x.tolist()

    face = torch.tensor([[0, 1], [1, 2], [2, 3]])
    data = Data(num_nodes=4, face=face)
    assert data.num_faces == 2
    assert data.num_nodes == 4

    data = Data(title="test")
    assert data.__repr__() == 'Data(title=test)'
    assert data.num_node_features == 0
    assert data.num_edge_features == 0

    torch_geometric.set_debug(False)


def test_debug_data():
    torch_geometric.set_debug(True)

    Data()
    Data(edge_index=torch.zeros((2, 0), dtype=torch.long), num_nodes=10)
    Data(face=torch.zeros((3, 0), dtype=torch.long), num_nodes=10)
    Data(edge_index=torch.tensor([[0, 1], [1, 0]]), edge_attr=torch.randn(2))
    Data(x=torch.torch.randn(5, 3), num_nodes=5)
    Data(pos=torch.torch.randn(5, 3), num_nodes=5)
    Data(norm=torch.torch.randn(5, 3), num_nodes=5)

    torch_geometric.set_debug(False)
