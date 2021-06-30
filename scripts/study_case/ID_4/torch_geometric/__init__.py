from .debug import is_debug_enabled, debug, set_debug
import scripts.study_case.ID_4.torch_geometric.nn
import scripts.study_case.ID_4.torch_geometric.data
import scripts.study_case.ID_4.torch_geometric.datasets
import scripts.study_case.ID_4.torch_geometric.transforms
import scripts.study_case.ID_4.torch_geometric.utils

__version__ = '1.3.0'

__all__ = [
    'is_debug_enabled',
    'debug',
    'set_debug',
    'torch_geometric',
    '__version__',
]
