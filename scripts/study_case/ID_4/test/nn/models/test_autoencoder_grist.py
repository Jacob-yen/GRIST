import torch
from torch import Tensor as T
import sys
sys.path.append("/data")
from scripts.study_case.ID_4.torch_geometric.nn import GAE, VGAE, ARGA, ARGVA
from scripts.study_case.ID_4.torch_geometric.data import Data


def test_gae():
    model = GAE(encoder=lambda x: x)
    model.reset_parameters()

    x = torch.Tensor([[1, -1], [1, 2], [2, 1]])
    z = model.encode(x)
    assert z.tolist() == x.tolist()

    adj = model.decoder.forward_all(z)
    assert adj.tolist() == torch.sigmoid(
        torch.Tensor([[+2, -1, +1], [-1, +5, +4], [+1, +4, +5]])).tolist()

    edge_index = torch.tensor([[0, 1], [1, 2]])
    value = model.decode(z, edge_index)
    assert value.tolist() == torch.sigmoid(torch.Tensor([-1, 4])).tolist()

    edge_index = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                               [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    data = Data(edge_index=edge_index)
    data.num_nodes = edge_index.max().item() + 1
    data = model.split_edges(data, val_ratio=0.2, test_ratio=0.3)

    assert data.val_pos_edge_index.size() == (2, 2)
    assert data.val_neg_edge_index.size() == (2, 2)
    assert data.test_pos_edge_index.size() == (2, 3)
    assert data.test_neg_edge_index.size() == (2, 3)
    assert data.train_pos_edge_index.size() == (2, 10)
    assert data.train_neg_adj_mask.size() == (11, 11)
    assert data.train_neg_adj_mask.sum().item() == (11**2 - 11) / 2 - 4 - 6 - 5

    z = torch.randn(11, 16)
    loss = model.recon_loss(z, data.train_pos_edge_index)
    assert loss.item() > 0

    auc, ap = model.test(z, data.val_pos_edge_index, data.val_neg_edge_index)
    assert auc >= 0 and auc <= 1 and ap >= 0 and ap <= 1


def test_vgae():
    from datetime import datetime
    import sys
    print("test vgae")
    model = VGAE(encoder=lambda x: (x, x))
    s1 = datetime.now()
    x = torch.Tensor([[1, -1], [1, 2], [2, 1]])

    '''inserted code'''
    from scripts.utils.torch_utils import GradientSearcher
    gradient_search = GradientSearcher(name="pytorch_geometric_fork_grist",switch_data=False,)
    gradient_search.build(batch_size=3)
    '''inserted code'''

    for i in range(10000):
        x = torch.autograd.Variable(x, requires_grad=True)
        model.encode(x)
        # assert model.kl_loss().item() > 0

        model.eval()
        model.encode(x)

        obj_function = -1 * torch.max(model.__logvar__)
        obj_grads = torch.autograd.grad(obj_function, x, retain_graph=True)[0]

        '''inserted code'''
        gradient_search.loss_checker(model.kl_loss())
        '''inserted code'''

        """inserted code"""
        monitor_vars = {'loss': model.kl_loss(), 'obj_function': obj_function, 'obj_grad': obj_grads}
        x , _ = gradient_search.update_batch_data( monitor_var=monitor_vars,input_data=x,clip=False)
        gradient_search.check_time()
        """inserted code"""


def test_arga():
    model = ARGA(encoder=lambda x: x, discriminator=lambda x: T([0.5]))
    model.reset_parameters()

    x = torch.Tensor([[1, -1], [1, 2], [2, 1]])
    z = model.encode(x)

    assert model.reg_loss(z).item() > 0
    assert model.discriminator_loss(z).item() > 0


def test_argva():
    model = ARGVA(encoder=lambda x: (x, x), discriminator=lambda x: T([0.5]))

    x = torch.Tensor([[1, -1], [1, 2], [2, 1]])
    model.encode(x)
    model.reparametrize(model.__mu__, model.__logvar__)
    assert model.kl_loss().item() > 0

test_vgae()