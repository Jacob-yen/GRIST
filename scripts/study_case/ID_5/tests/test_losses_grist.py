import torch
import numpy as np
import sys
sys.path.append("/data")

from scripts.study_case.ID_5.matchzoo import losses


def test_hinge_loss():
    true_value = torch.Tensor([[1.2], [1], [1], [1]])
    pred_value = torch.Tensor([[1.2], [0.1], [0], [-0.3]])
    expected_loss = torch.Tensor([(0 + 1 - 0.3 + 0) / 2.0])
    loss = losses.RankHingeLoss()(pred_value, true_value)
    assert torch.isclose(expected_loss, loss)
    expected_loss = torch.Tensor(
        [(2 + 0.1 - 1.2 + 2 - 0.3 + 0) / 2.0])
    loss = losses.RankHingeLoss(margin=2)(pred_value, true_value)
    assert torch.isclose(expected_loss, loss)
    true_value = torch.Tensor(
        [[1.2], [1], [0.8], [1], [1], [0.8]])
    pred_value = torch.Tensor(
        [[1.2], [0.1], [-0.5], [0], [0], [-0.3]])
    expected_loss = torch.Tensor(
        [(0 + 1 - 0.15) / 2.0])
    loss = losses.RankHingeLoss(num_neg=2, margin=1)(
        pred_value, true_value)
    assert torch.isclose(expected_loss, loss)

# def test_rank_crossentropy_loss():
#     from datetime import datetime
#     import sys
#
#     losses.neg_num = 1
#
#     true_value = torch.Tensor([[1.], [0.], [0.], [0.], [1.], [0.]])
#     pred_value = torch.Tensor([[0.8], [0.1], [0.1], [0.8], [0.1], [0.1]])
#
#     """insert code"""
#     from scripts.utils.torch_utils import GradientSearcher
#     gradient_search = GradientSearcher(name="MatchZoo_grist")
#     max_val, min_val = 1, 0
#     gradient_search.build(batch_size=1, min_val=min_val, max_val=max_val)
#     """insert code"""
#
#     while True:
#         pred_value = torch.autograd.Variable(pred_value, requires_grad=True)
#         loss,obj_var = losses.RankCrossEntropyLossFGSM(num_neg=2)(
#             pred_value, true_value)
#
#         """inserted code"""
#         obj_function = torch.min(torch.abs(obj_var))
#         obj_grads = torch.autograd.grad(obj_function, pred_value, retain_graph=True)[0]
#         monitor_vars = {'loss': loss, 'obj_function': obj_function, 'obj_grad': obj_grads}
#         pred_value, _ = gradient_search.update_batch_data(monitor_var=monitor_vars, input_data=pred_value,switch_data=False)
#         """inserted code"""

def test_rank_crossentropy_loss():
    from datetime import datetime
    import sys

    s1 = datetime.now()
    print("test")
    losses.neg_num = 1

    true_value = torch.Tensor([[1.], [0.], [0.], [0.], [1.], [0.]])
    pred_value = torch.Tensor([[0.8], [0.1], [0.1], [0.8], [0.1], [0.1]])

    '''inserted code'''
    from scripts.utils.torch_utils import GradientSearcher
    gradient_search = GradientSearcher(name="Matchzoo_grist",switch_data=False)
    gradient_search.build(batch_size=1)
    '''inserted code'''

    while True:
        pred_value = torch.autograd.Variable(pred_value, requires_grad=True)
        loss,obj_var = losses.RankCrossEntropyLossFGSM(num_neg=2)(
            pred_value, true_value)
        obj_function = torch.min(torch.abs(obj_var))
        obj_grads = torch.autograd.grad(obj_function, pred_value, retain_graph=True)[0]

        """inserted code"""
        monitor_vars = {'loss': loss, 'obj_function': obj_function, 'obj_grad': obj_grads}
        pred_value, _ = gradient_search.update_batch_data( monitor_var=monitor_vars,input_data=pred_value,clip=False)
        """inserted code"""


        # obj_grads[obj_grads > 0] = 1
        # obj_grads[obj_grads < 0] = -1
        # obj_grads = obj_grads * 0.1
        #
        # pred_value = pred_value - obj_grads

        # pred_value = torch.clamp(pred_value,0,1)

        '''inserted code'''
        gradient_search.check_time()
        '''inserted code'''

test_rank_crossentropy_loss()
