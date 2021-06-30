from scipy.sparse.linalg import eigs, eigsh
from scripts.study_case.ID_4.torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix


class LaplacianLambdaMax(object):
    r"""Computes the highest eigenvalue of the graph Laplacian given by
    :meth:`torch_geometric.utils.get_laplacian`.

    Args:
        normalization (str, optional): The normalization scheme for the graph
            Laplacian (default: :obj:`None`):

            1. :obj:`None`: No normalization
            :math:`\mathbf{L} = \mathbf{D} - \mathbf{A}`

            2. :obj:`"sym"`: Symmetric normalization
            :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1/2} \mathbf{A}
            \mathbf{D}^{-1/2}`

            3. :obj:`"rw"`: Random-walk normalization
            :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1} \mathbf{A}`
    """

    def __init__(self, normalization=None):
        assert normalization in [None, 'sym', 'rw'], 'Invalid normalization'
        self.normalization = normalization

    def __call__(self, data):
        edge_weight = data.edge_attr
        if edge_weight is not None and edge_weight.numel() != data.num_edges:
            edge_weight = None

        edge_index, edge_weight = get_laplacian(data.edge_index, edge_weight,
                                                self.normalization,
                                                num_nodes=data.num_nodes)

        L = to_scipy_sparse_matrix(edge_index, edge_weight, data.num_nodes)

        eig_fn = eigsh if self.normalization == 'sym' else eigs
        lambda_max = eig_fn(L, k=1, which='LM', return_eigenvectors=False)
        data.lambda_max = float(lambda_max.real)

        return data

    def __repr__(self):
        return '{}(normalization={})'.format(self.__class__.__name__,
                                             self.normalization)
