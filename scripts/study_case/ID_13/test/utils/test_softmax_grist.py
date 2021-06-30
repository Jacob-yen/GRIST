from math import exp

import torch
import sys
sys.path.append("/data")
from torch.autograd import Variable
from scripts.study_case.ID_13.torch_geometric.utils import softmax
from datetime import datetime
import sys

def test_softmax():
    row = torch.LongTensor([0, 0, 0, 1, 1, 1])
    edge_attr = torch.Tensor([0, 1, 2, 0, 1, 2])
    '''inserted code'''
    from scripts.utils.torch_utils import GradientSearcher
    gradient_search = GradientSearcher(name="pytorch_geometric_exp_grist",switch_data=False,)
    gradient_search.build(batch_size=1)
    '''inserted code'''

    while True:
        edge_attr = Variable(edge_attr, requires_grad=True)
        out = softmax(edge_attr, row)
        obj_function = -1 * torch.max(edge_attr)
        obj_grads = torch.autograd.grad(obj_function, edge_attr, retain_graph=True)[0]

        '''inserted code'''
        gradient_search.loss_checker(out)
        '''inserted code'''

        """inserted code"""
        monitor_vars = {'loss': out, 'obj_function': obj_function, 'obj_grad': obj_grads}
        edge_attr, _ = gradient_search.update_batch_data(monitor_var=monitor_vars, input_data=edge_attr,clip=False)
        """inserted code"""

    # output = softmax(edge_attr, row)

    # output = softmax(Variable(edge_attr), row)
    #
    # output = softmax(edge_attr, col)
    #
    # edge_attr = torch.Tensor([[1, 1], [2, 2], [3, 3], [1, 1], [2, 2], [3, 3]])
    # output = softmax(edge_attr, row)
    #
    # output = softmax(Variable(edge_attr), row)
    # # output
test_softmax()