import torch
import scripts.study_case.ID_4.torch_geometric
from scripts.study_case.ID_4.torch_geometric.data import Data


class Batch(Data):
    r"""A plain old python object modeling a batch of graphs as one big
    (dicconnected) graph. With :class:`torch_geometric.data.Data` being the
    base class, all its methods can also be used here.
    In addition, single graphs can be reconstructed via the assignment vector
    :obj:`batch`, which maps each node to its respective graph identifier.
    """

    def __init__(self, batch=None, **kwargs):
        super(Batch, self).__init__(**kwargs)

        self.batch = batch
        self.__data_class__ = Data
        self.__slices__ = None

    @staticmethod
    def from_data_list(data_list, follow_batch=[]):
        r"""Constructs a batch object from a python list holding
        :class:`torch_geometric.data.Data` objects.
        The assignment vector :obj:`batch` is created on the fly.
        Additionally, creates assignment batch vectors for each key in
        :obj:`follow_batch`."""

        keys = [set(data.keys) for data in data_list]
        keys = list(set.union(*keys))
        assert 'batch' not in keys

        batch = Batch()
        batch.__data_class__ = data_list[0].__class__
        batch.__slices__ = {key: [0] for key in keys}

        for key in keys:
            batch[key] = []

        for key in follow_batch:
            batch['{}_batch'.format(key)] = []

        cumsum = {key: 0 for key in keys}
        batch.batch = []
        for i, data in enumerate(data_list):
            for key in data.keys:
                item = data[key] + cumsum[key]
                if torch.is_tensor(data[key]):
                    size = data[key].size(data.__cat_dim__(key, data[key]))
                else:
                    size = 1
                batch.__slices__[key].append(size + batch.__slices__[key][-1])
                cumsum[key] += data.__inc__(key, item)
                batch[key].append(item)

                if key in follow_batch:
                    item = torch.full((size, ), i, dtype=torch.long)
                    batch['{}_batch'.format(key)].append(item)

            num_nodes = data.num_nodes
            if num_nodes is not None:
                item = torch.full((num_nodes, ), i, dtype=torch.long)
                batch.batch.append(item)

        if num_nodes is None:
            batch.batch = None

        for key in batch.keys:
            item = batch[key][0]
            if torch.is_tensor(item):
                batch[key] = torch.cat(batch[key],
                                       dim=data_list[0].__cat_dim__(key, item))
            elif isinstance(item, int) or isinstance(item, float):
                batch[key] = torch.tensor(batch[key])
            else:
                raise ValueError('Unsupported attribute type')

        # Copy custom data functions to batch (does not work yet):
        # if data_list.__class__ != Data:
        #     org_funcs = set(Data.__dict__.keys())
        #     funcs = set(data_list[0].__class__.__dict__.keys())
        #     batch.__custom_funcs__ = funcs.difference(org_funcs)
        #     for func in funcs.difference(org_funcs):
        #         setattr(batch, func, getattr(data_list[0], func))

        if torch_geometric.is_debug_enabled():
            batch.debug()

        return batch.contiguous()

    def to_data_list(self):
        r"""Reconstructs the list of :class:`torch_geometric.data.Data` objects
        from the batch object.
        The batch object must have been created via :meth:`from_data_list` in
        order to be able reconstruct the initial objects."""

        if self.__slices__ is None:
            raise RuntimeError(
                ('Cannot reconstruct data list from batch because the batch '
                 'object was not created using Batch.from_data_list()'))

        keys = [key for key in self.keys if key[-5:] != 'batch']
        cumsum = {key: 0 for key in keys}
        data_list = []
        for i in range(len(self.__slices__[keys[0]]) - 1):
            data = self.__data_class__()
            for key in keys:
                data[key] = self[key].narrow(
                    data.__cat_dim__(key, self[key]), self.__slices__[key][i],
                    self.__slices__[key][i + 1] - self.__slices__[key][i])
                data[key] = data[key] - cumsum[key]
                cumsum[key] += data.__inc__(key, data[key])
            data_list.append(data)

        return data_list

    @property
    def num_graphs(self):
        """Returns the number of graphs in the batch."""
        return self.batch[-1].item() + 1


class GraphLevelBatch(Batch):
    def __init__(self, batch=None, **kwargs):
        super(GraphLevelBatch, self).__init__(batch, **kwargs)

    @staticmethod
    def from_graph_data_list(data_list, follow_batch=[]):
        r"""Constructs a batch object from a python list holding
        :class:`torch_geometric.data.Data` objects.
        The assignment vector :obj:`batch` is created on the fly."""
        keys = [set(data.keys) for data in data_list]
        keys = list(set.union(*keys))
        assert 'batch' not in keys

        batch = Batch()

        for key in keys:
            batch[key] = []

        for key in follow_batch:
            batch['{}_batch'.format(key)] = []

        batch.batch = []
        cumsum = 0

        hierarchy_cumsum = [0 for _ in range(len(data_list[0].num_vertices))]
        valid_cumsum = 0

        for i, data in enumerate(data_list):
            num_nodes = data.num_nodes
            batch.batch.append(torch.full((num_nodes, ), i, dtype=torch.long))
            for key in data.keys:
                item = data[key]

                if 'hierarchy' in key:
                    level = int(key[-1]) - 1
                    item = item + hierarchy_cumsum[level] if data.__cumsum__(key, item) else item
                elif key.startswith('original_index'):
                    if type(item) != list:
                        item = item + valid_cumsum if data.__cumsum__(key, item) else item
                    else:
                        item[0] = item[0] + valid_cumsum if data.__cumsum__(key, item[0]) else item[0]
                        for i in range(1, len(item)):
                            item[i] = item[i] + hierarchy_cumsum[i-1] if data.__cumsum__(key, item[i]) else item[i]
                elif key.startswith('full_mesh_index_recover'):
                    pass
                elif key.startswith('full_mesh_labels'):
                    pass
                else:
                    item = item + cumsum if data.__cumsum__(key, item) else item

                batch[key].append(item)

            for key in follow_batch:
                size = data[key].size(data.__cat_dim__(key, data[key]))
                item = torch.full((size, ), i, dtype=torch.long)
                batch['{}_batch'.format(key)].append(item)

            cumsum += num_nodes

            for i in range(len(data.num_vertices)):
                hierarchy_cumsum[i] += data.num_vertices[i]

            valid_cumsum += num_nodes

        for key in keys:
            item = batch[key][0]
            if key.startswith('full_mesh_index_recover'):
                pass
            elif key.startswith('full_mesh_labels'):
                pass
            elif torch.is_tensor(item):
                batch[key] = torch.cat(
                    batch[key], dim=data_list[0].__cat_dim__(key, item))
            elif isinstance(item, int) or isinstance(item, float):
                batch[key] = torch.tensor(batch[key])
            else:
                pass
        batch.batch = torch.cat(batch.batch, dim=-1)
        return batch.contiguous()


