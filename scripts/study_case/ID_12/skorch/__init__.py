"""skorch base imports"""

import pkg_resources
from pkg_resources import parse_version

from .history import History
from .net import NeuralNet
# from .net_fgsm import NeuralNetFGSM
from .classifier import NeuralNetClassifier
from .classifier import NeuralNetBinaryClassifier
# from .classifier_fgsm import NeuralNetClassifierFGSM
# from .classifier_fgsm import NeuralNetBinaryClassifierFGSM
from .regressor import NeuralNetRegressor
from . import callbacks


MIN_TORCH_VERSION = '0.4.1'

try:
    # pylint: disable=wrong-import-position
    import torch
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "No module named 'torch', and skorch depends on PyTorch "
        "(aka 'torch'). "
        "Visit https://pytorch.org/ for installation instructions.")

torch_version = pkg_resources.get_distribution('torch').version
if parse_version(torch_version) < parse_version(MIN_TORCH_VERSION):
    msg = ('skorch depends on a newer version of PyTorch (at least {req}, not '
           '{installed}). Visit https://pytorch.org for installation details')
    raise ImportWarning(msg.format(req=MIN_TORCH_VERSION, installed=torch_version))


__all__ = [
    'History',
    'NeuralNet',
    # 'NeuralNetFGSM',
    'NeuralNetClassifier',
    'NeuralNetBinaryClassifier',
    # 'NeuralNetClassifierFGSM',
    # 'NeuralNetBinaryClassifierFGSM',
    'NeuralNetRegressor',
    'callbacks',
]


try:
    __version__ = pkg_resources.get_distribution('skorch').version
except:  # pylint: disable=bare-except
    __version__ = 'n/a'
