*This is our fork of PyTorch Geometric for enabling hierarchical mesh structures*

[pypi-image]: https://badge.fury.io/py/torch-geometric.svg
[pypi-url]: https://pypi.python.org/pypi/torch-geometric
[build-image]: https://travis-ci.org/rusty1s/pytorch_geometric.svg?branch=master
[build-url]: https://travis-ci.org/rusty1s/pytorch_geometric
[docs-image]: https://readthedocs.org/projects/pytorch-geometric/badge/?version=latest
[docs-url]: https://pytorch-geometric.readthedocs.io/en/latest/?badge=latest
[coverage-image]: https://codecov.io/gh/rusty1s/pytorch_geometric/branch/master/graph/badge.svg
[coverage-url]: https://codecov.io/github/rusty1s/pytorch_geometric?branch=master
[contributing-image]: https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat
[contributing-url]: https://github.com/rusty1s/pytorch_geometric/blob/master/CONTRIBUTING.md

<p align="center">
  <img width="40%" src="https://raw.githubusercontent.com/rusty1s/pytorch_geometric/master/docs/source/_static/img/pyg_logo_text.svg?sanitize=true" />
</p>

--------------------------------------------------------------------------------

[![PyPI Version][pypi-image]][pypi-url]
[![Build Status][build-image]][build-url]
[![Docs Status][docs-image]][docs-url]
[![Code Coverage][coverage-image]][coverage-url]
[![Contributing][contributing-image]][contributing-url]

**[Documentation](https://pytorch-geometric.readthedocs.io)** | **[Paper](https://arxiv.org/abs/1903.02428)** | **[External Resources](https://pytorch-geometric.readthedocs.io/en/latest/notes/resources.html)**

*PyTorch Geometric* (PyG) is a geometric deep learning extension library for [PyTorch](https://pytorch.org/).

It consists of various methods for deep learning on graphs and other irregular structures, also known as *[geometric deep learning](http://geometricdeeplearning.com/)*, from a variety of published papers.
In addition, it consists of an easy-to-use mini-batch loader for many small and single giant graphs, multi gpu-support, a large number of common benchmark datasets (based on simple interfaces to create your own), and helpful transforms, both for learning on arbitrary graphs as well as on 3D meshes or point clouds.

--------------------------------------------------------------------------------

PyTorch Geometric makes implementing Graph Neural Networks a breeze (see [here](https://pytorch-geometric.readthedocs.io/en/latest/notes/create_gnn.html) for the accompanying tutorial).
For example, this is all it takes to implement the [edge convolutional layer](https://arxiv.org/abs/1801.07829):

```python
import torch
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from scripts.study_case.ID_4.torch_geometric.nn import MessagePassing

class EdgeConv(MessagePassing):
    def __init__(self, F_in, F_out):
        super(EdgeConv, self).__init__(aggr='max')  # "Max" aggregation.
        self.mlp = Seq(Lin(2 * F_in, F_out), ReLU(), Lin(F_out, F_out))

    def forward(self, x, edge_index):
        # x has shape [N, F_in]
        # edge_index has shape [2, E]
        return self.propagate(edge_index, x=x)  # shape [N, F_out]

    def message(self, x_i, x_j):
        # x_i has shape [E, F_in]
        # x_j has shape [E, F_in]
        edge_features = torch.cat([x_i, x_j - x_i], dim=1)  # shape [E, 2 * F_in]
        return self.mlp(edge_features)  # shape [E, F_out]
```

In detail, the following methods are currently implemented:

* **[SplineConv](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.SplineConv)** from Fey *et al.*: [SplineCNN: Fast Geometric Deep Learning with Continuous B-Spline Kernels](https://arxiv.org/abs/1711.08920) (CVPR 2018)
* **[GCNConv](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GCNConv)** from Kipf and Welling: [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907) (ICLR 2017)
* **[ChebConv](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.ChebConv)** from Defferrard *et al.*: [Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering](https://arxiv.org/abs/1606.09375) (NIPS 2016)
* **[NNConv](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.NNConv)** from Gilmer *et al.*: [Neural Message Passing for Quantum Chemistry](https://arxiv.org/abs/1704.01212) (ICML 2017)
* **[ECConv](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.ECConv)** from Simonovsky and Komodakis: [Edge-Conditioned Convolution on Graphs](https://arxiv.org/abs/1704.02901) (CVPR 2017)
* **[GATConv](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GATConv)** from Veličković *et al.*: [Graph Attention Networks](https://arxiv.org/abs/1710.10903) (ICLR 2018)
* **[SAGEConv](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.SAGEConv)** from Hamilton *et al.*: [Inductive Representation Learning on Large Graphs](https://arxiv.org/abs/1706.02216) (NIPS 2017)
* **[GraphConv](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GraphConv)** from, *e.g.*, Morris *et al.*: [Weisfeiler and Leman Go Neural: Higher-order Graph Neural Networks](https://arxiv.org/abs/1810.02244) (AAAI 2019)
* **[GatedGraphConv](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GatedGraphConv)** from Li *et al.*: [Gated Graph Sequence Neural Networks](https://arxiv.org/abs/1511.05493) (ICLR 2016)
* **[GINConv](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GINConv)** from Xu *et al.*: [How Powerful are Graph Neural Networks?](https://arxiv.org/abs/1810.00826) (ICLR 2019)
* **[ARMAConv](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.ARMAConv)** from Bianchi *et al.*: [Graph Neural Networks with Convolutional ARMA Filters](https://arxiv.org/abs/1901.01343) (CoRR 2019)
* **[SGConv](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.SGConv)** from Wu *et al.*: [Simplifying Graph Convolutional Networks](https://arxiv.org/abs/1902.07153) (CoRR 2019)
* **[APPNP](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.APPNP)** from Klicpera *et al.*: [Predict then Propagate: Graph Neural Networks meet Personalized PageRank](https://arxiv.org/abs/1810.05997) (ICLR 2019)
* **[AGNNConv](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.AGNNConv)** from Thekumparampil *et al.*: [Attention-based Graph Neural Network for Semi-Supervised Learning](https://arxiv.org/abs/1803.03735) (CoRR 2017)
* **[TAGConv](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.TAGConv)** from Du *et al.*: [Topology Adaptive Graph Convolutional Networks](https://arxiv.org/abs/1710.10370) (CoRR 2017)
* **[RGCNConv](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.RGCNConv)** from Schlichtkrull *et al.*: [Modeling Relational Data with Graph Convolutional Networks](https://arxiv.org/abs/1703.06103) (ESWC 2018)
* **[SignedConv](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.SignedConv)** from Derr *et al.*: [Signed Graph Convolutional Network](https://arxiv.org/abs/1808.06354) (ICDM 2018)
* **[DNAConv](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.DNAConv)** from Fey: [Just Jump: Dynamic Neighborhood Aggregation in Graph Neural Networks](https://arxiv.org/abs/1904.04849) (ICLR-W 2019)
* **[EdgeConv](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.EdgeConv)** from Wang *et al.*: [Dynamic Graph CNN for Learning on Point Clouds](https://arxiv.org/abs/1801.07829) (CoRR, 2018)
* **[PointConv](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.PointConv)** (including **[Iterative Farthest Point Sampling](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.pool.fps)**, dynamic graph generation based on **[nearest neighbor](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.pool.knn_graph)** or **[maximum distance](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.pool.radius_graph)**, and **[k-NN interpolation](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.unpool.knn_interpolate)** for upsampling) from Qi *et al.*: [PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation](https://arxiv.org/abs/1612.00593) (CVPR 2017) and [PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space](https://arxiv.org/abs/1706.02413) (NIPS 2017)
* **[XConv](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.XConv)** from Li *et al.*: [PointCNN: Convolution On X-Transformed Points](https://arxiv.org/abs/1801.07791) [(official implementation)](https://github.com/yangyanli/PointCNN) (NeurIPS 2018)
* **[PPFConv](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.PPFConv)** from Deng *et al.*: [PPFNet: Global Context Aware Local Features for Robust 3D Point Matching](https://arxiv.org/abs/1802.02669) (CVPR 2018)
* **[GMMConv](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GMMConv)** from Monti *et al.*: [Geometric Deep Learning on Graphs and Manifolds using Mixture Model CNNs](https://arxiv.org/abs/1611.08402) (CVPR 2017)
* **[FeaStConv](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.FeaStConv)** from Verma *et al.*: [FeaStNet: Feature-Steered Graph Convolutions for 3D Shape Analysis](https://arxiv.org/abs/1706.05206) (CVPR 2018)
* **[HypergraphConv](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.HypergraphConv)** from Bai *et al.*: [Hypergraph Convolution and Hypergraph Attention](https://arxiv.org/abs/1901.08150) (CoRR 2019)
* A **[MetaLayer](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.meta.MetaLayer)** for building any kind of graph network similar to the [TensorFlow Graph Nets library](https://github.com/deepmind/graph_nets) from Battaglia *et al.*: [Relational Inductive Biases, Deep Learning, and Graph Networks](https://arxiv.org/abs/1806.01261) (CoRR 2018)
* **[GlobalAttention](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.glob.GlobalAttention)** from Li *et al.*: [Gated Graph Sequence Neural Networks](https://arxiv.org/abs/1511.05493) (ICLR 2016)
* **[Set2Set](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.glob.Set2Set)** from Vinyals *et al.*: [Order Matters: Sequence to Sequence for Sets](https://arxiv.org/abs/1511.06391) (ICLR 2016)
* **[Sort Pool](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.glob.global_sort_pool)** from Zhang *et al.*: [An End-to-End Deep Learning Architecture for Graph Classification](https://www.cse.wustl.edu/~muhan/papers/AAAI_2018_DGCNN.pdf) (AAAI 2018)
* **[Dense Differentiable Pooling](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.dense.diff_pool.dense_diff_pool)** from Ying *et al.*: [Hierarchical Graph Representation Learning with Differentiable Pooling](https://arxiv.org/abs/1806.08804) (NeurIPS 2018)
* **[Graclus Pooling](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.pool.graclus)** from Dhillon *et al.*: [Weighted Graph Cuts without Eigenvectors: A Multilevel Approach](http://www.cs.utexas.edu/users/inderjit/public_papers/multilevel_pami.pdf) (PAMI 2007)
* **[Voxel Grid Pooling](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.pool.voxel_grid)** from, *e.g.*, Simonovsky and Komodakis: [Dynamic Edge-Conditioned Filters in Convolutional Neural Networks on Graphs](https://arxiv.org/abs/1704.02901) (CVPR 2017)
* **[Top-K Pooling](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.pool.TopKPooling)** from Gao and Ji: [Graph U-Nets](https://arxiv.org/abs/1905.05178) (ICML 2019), Cangea *et al.*: [Towards Sparse Hierarchical Graph Classifiers](https://arxiv.org/abs/1811.01287) (NeurIPS-W 2018) and Knyazev *et al.*: [Understanding Attention and Generalization in Graph Neural Networks](https://arxiv.org/abs/1905.02850) (ICLR-W 2019)
* **[SAG Pooling](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.pool.SAGPooling)** from Lee *et al.*: [Self-Attention Graph Pooling](https://arxiv.org/abs/1904.08082) (ICML 2019) and Knyazev *et al.*: [Understanding Attention and Generalization in Graph Neural Networks](https://arxiv.org/abs/1905.02850) (ICLR-W 2019)
* **[Edge Pooling](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.pool.EdgePooling)** from Diehl *et al.*: [Towards Graph Pooling by Edge Contraction](https://graphreason.github.io/papers/17.pdf) (ICML-W 2019) and Diehl: [Edge Contraction Pooling for Graph Neural Networks](https://arxiv.org/abs/1905.10990) (CoRR 2019)
* **[Local Degree Profile](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.transforms.LocalDegreeProfile)** from Cai and Wang: [A Simple yet Effective Baseline for Non-attribute Graph Classification](https://arxiv.org/abs/1811.03508) (CoRR 2018)
* **[Jumping Knowledge](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.models.JumpingKnowledge)** from Xu *et al.*: [Representation Learning on Graphs with Jumping Knowledge Networks](https://arxiv.org/abs/1806.03536) (ICML 2018)
* **[Node2Vec](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.models.Node2Vec)** from Grover and Leskovec: [node2vec: Scalable Feature Learning for Networks](https://arxiv.org/abs/1607.00653) (KDD 2016)
* **[Deep Graph Infomax](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.models.DeepGraphInfomax)** from Veličković *et al.*: [Deep Graph Infomax](https://arxiv.org/abs/1809.10341) (ICLR 2019)
* All variants of **[Graph Auto-Encoders](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.models.GAE)** from Kipf and Welling: [Variational Graph Auto-Encoders](https://arxiv.org/abs/1611.07308) (NIPS-W 2016) and Pan *et al.*: [Adversarially Regularized Graph Autoencoder for Graph Embedding](https://arxiv.org/abs/1802.04407) (IJCAI 2018)
* **[RENet](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.models.RENet)** from Jin *et al.*: [Recurrent Event Network for Reasoning over Temporal Knowledge Graphs](https://arxiv.org/abs/1904.05530) (ICLR-W 2019)
* **[GraphUNet](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.models.GraphUNet)** from Gao and Ji: [Graph U-Nets](https://arxiv.org/abs/1905.05178) (ICML 2019)
* **[NeighborSampler](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.data.NeighborSampler)** from Hamilton *et al.*: [Inductive Representation Learning on Large Graphs](https://arxiv.org/abs/1706.02216) (NIPS 2017)

--------------------------------------------------------------------------------

Head over to our [documentation](https://pytorch-geometric.readthedocs.io) to find out more about installation, data handling, creation of datasets and a full list of implemented methods, transforms, and datasets.
For a quick start, check out our [examples](https://github.com/rusty1s/pytorch_geometric/tree/master/examples) in the `examples/` directory.

If you notice anything unexpected, please open an [issue](https://github.com/rusty1s/pytorch_geometric/issues) and let us know.
If you are missing a specific method, feel free to open a [feature request](https://github.com/rusty1s/pytorch_geometric/issues).
We are motivated to constantly make PyTorch Geometric even better.

## Installation

Ensure that at least PyTorch 1.1.0 is installed and verify that `cuda/bin`, `cuda/include` and `cuda/lib64` are in your `$PATH`, `$CPATH` and `$LD_LIBRARY_PATH` respectively, *e.g.*:

```
$ python -c "import torch; print(torch.__version__)"
>>> 1.1.0

$ echo $PATH
>>> /usr/local/cuda/bin:...

$ echo $CPATH
>>> /usr/local/cuda/include:...
```
and
```
$ echo $LD_LIBRARY_PATH
>>> /usr/local/cuda/lib64
```
on Linux or
```
$ echo $DYLD_LIBRARY_PATH
>>> /usr/local/cuda/lib
```
on macOS.
Then run:

```sh
$ pip install --verbose --no-cache-dir torch-scatter
$ pip install --verbose --no-cache-dir torch-sparse
$ pip install --verbose --no-cache-dir torch-cluster
$ pip install --verbose --no-cache-dir torch-spline-conv (optional)
$ pip install torch-geometric
```

See the [Frequently Asked Questions](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html#frequently-asked-questions) and verify that your [CUDA is set up correctly](https://docs.nvidia.com/cuda/index.html) if you encounter any installation errors.

## Running examples

```
cd examples
python gcn.py
```

## Cite

Please cite [our paper](https://arxiv.org/abs/1903.02428) (and the respective papers of the methods used) if you use this code in your own work:

```
@inproceedings{Fey/Lenssen/2019,
  title={Fast Graph Representation Learning with {PyTorch Geometric}},
  author={Fey, Matthias and Lenssen, Jan E.},
  booktitle={ICLR Workshop on Representation Learning on Graphs and Manifolds},
  year={2019},
}
```

Feel free to [email us](mailto:matthias.fey@tu-dortmund.de) if you wish your work to be listed in the [external resources](https://pytorch-geometric.readthedocs.io/en/latest/notes/resources.html).

## Running tests

```
python setup.py test
```
