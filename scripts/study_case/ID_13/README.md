[pypi-image]: https://badge.fury.io/py/torch-geometric.svg
[pypi-url]: https://pypi.python.org/pypi/torch-geometric
[build-image]: https://travis-ci.org/rusty1s/pytorch_geometric.svg?branch=master
[build-url]: https://travis-ci.org/rusty1s/pytorch_geometric
[coverage-image]: https://codecov.io/gh/rusty1s/pytorch_geometric/branch/master/graph/badge.svg
[coverage-url]: https://codecov.io/github/rusty1s/pytorch_geometric?branch=master

<p align="center">
  <img width="40%" src="https://raw.githubusercontent.com/rusty1s/pytorch_geometric/master/docs/source/_static/img/logo.svg?sanitize=true" />
</p>

--------------------------------------------------------------------------------

[![PyPI Version][pypi-image]][pypi-url]
[![Build Status][build-image]][build-url]
[![Code Coverage][coverage-image]][coverage-url]

**[Documentation](http://rusty1s.github.io/pytorch_geometric)**

*PyTorch Geometric* is a geometric deep learning extension library for [PyTorch](https://pytorch.org/).

It consists of various methods for deep learning on graphs and other irregular structures, also known as *[geometric deep learning](http://geometricdeeplearning.com/)*, from a variety of published papers.
In addition, it consists of an easy-to-use mini-batch loader, a large number of common benchmark datasets (based on simple interfaces to create your own), and helpful transforms, both for learning on arbitrary graphs as well as on 3D meshes or point clouds.

In detail, the following methods are currently implemented:

* **[SplineConv](https://rusty1s.github.io/pytorch_geometric/build/html/modules/nn.html#torch_geometric.nn.conv.SplineConv)** from Fey *et al.*: [SplineCNN: Fast Geometric Deep Learning with Continuous B-Spline Kernels](https://arxiv.org/abs/1711.08920) (CVPR 2018)
* **[GCNConv](https://rusty1s.github.io/pytorch_geometric/build/html/modules/nn.html#torch_geometric.nn.conv.GCNConv)** from Kipf and Welling: [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907) (ICLR 2017)
* **[ChebConv](https://rusty1s.github.io/pytorch_geometric/build/html/modules/nn.html#torch_geometric.nn.conv.ChebConv)** from Defferrard *et al.*: [Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering](https://arxiv.org/abs/1606.09375) (NIPS 2016)
* **[NNConv](https://rusty1s.github.io/pytorch_geometric/build/html/modules/nn.html#torch_geometric.nn.conv.NNConv)** adapted from Gilmer *et al.*: [Neural Message Passing for Quantum Chemistry](https://arxiv.org/abs/1704.01212) (ICML 2017)
* **[GATConv](https://rusty1s.github.io/pytorch_geometric/build/html/modules/nn.html#torch_geometric.nn.conv.GATConv)** from Veličković *et al.*: [Graph Attention Networks](https://arxiv.org/abs/1710.10903) (ICLR 2018)
* **[AGNNProp](https://rusty1s.github.io/pytorch_geometric/build/html/modules/nn.html#torch_geometric.nn.prop.AGNNProp)** from Kiran *et al.*: [Attention-based Graph Neural Network for Semi-Supervised Learning](https://arxiv.org/abs/1803.03735) (CoRR)
* **[SAGEMeanConv](https://rusty1s.github.io/pytorch_geometric/build/html/modules/nn.html#torch_geometric.nn.conv.SAGEMeanConv)** from Hamilton *et al.*: [Inductive Representation Learning on Large Graphs](https://arxiv.org/abs/1706.02216) (NIPS 2017)
* **[GINConv](https://rusty1s.github.io/pytorch_geometric/build/html/modules/nn.html#torch_geometric.nn.conv.GINConv)** from Xu *et al.*: [How Powerful are Graph Neural Networks?](https://arxiv.org/abs/1810.00826) (ICLR 2019, under review)
* A **[MetaLayer](https://rusty1s.github.io/pytorch_geometric/build/html/modules/nn.html#torch_geometric.nn.MetaLayer)** for building any kind of graph network from Battaglia *et al.*: [Relational Inductive Biases, Deep Learning, and Graph Networks](https://arxiv.org/abs/1806.01261) (CoRR)
* **[Set2Set](https://rusty1s.github.io/pytorch_geometric/build/html/modules/nn.html#torch_geometric.nn.pool.Set2Set)** from Vinyals *et al.*: [Order Matters: Sequence to Sequence for Sets](https://arxiv.org/abs/1511.06391) (ICLR 2016)
* **[Dense Differentiable Pooling](https://rusty1s.github.io/pytorch_geometric/build/html/modules/nn.html#torch_geometric.nn.dense.dense_diff_pool)** from Ying *et al.*: [Hierarchical Graph Representation Learning with Differentiable Pooling](https://arxiv.org/abs/1806.08804) (NIPS 2018)
* **[Graclus Pooling](https://rusty1s.github.io/pytorch_geometric/build/html/modules/nn.html#torch_geometric.nn.pool.graclus)** from Dhillon *et al.*: [Weighted Graph Cuts without Eigenvectors: A Multilevel Approach](http://www.cs.utexas.edu/users/inderjit/public_papers/multilevel_pami.pdf) (PAMI 2007)
* **[Voxel Grid Pooling](https://rusty1s.github.io/pytorch_geometric/build/html/modules/nn.html#torch_geometric.nn.pool.voxel_grid)** from, *e.g.*, Simonovsky and Komodakis: [Dynamic Edge-Conditioned Filters in Convolutional Neural Networks on Graphs](https://arxiv.org/abs/1704.02901) (CVPR 2017)
* **[Top-K Pooling](https://rusty1s.github.io/pytorch_geometric/build/html/modules/nn.html#torch_geometric.nn.pool.TopKPooling)** from Anonymous: [Graph U-Net](https://openreview.net/forum?id=HJePRoAct7) (ICLR 2019, under review) and Cangea *et al.*: [Towards Sparse Hierarchical Graph Classifiers](https://arxiv.org/abs/1811.01287) (CoRR)
* **[Sort Pool](https://rusty1s.github.io/pytorch_geometric/build/html/modules/nn.html#torch_geometric.nn.pool.sort_pool)** from Zhang *et al.*: [An End-to-End Deep Learning Architecture for Graph Classification](https://www.cse.wustl.edu/~muhan/papers/AAAI_2018_DGCNN.pdf) (AAAI 2018)
* Example of **[Deep Graph Infomax on Cora](https://github.com/rusty1s/pytorch_geometric/tree/master/examples/infomax.py)** from Veličković *et al.*: [Deep Graph Infomax](https://arxiv.org/abs/1809.10341) (ICLR 2019, under review)

Head over to our [documentation](http://rusty1s.github.io/pytorch_geometric) to find out more about installation, data handling, creation of datasets and a full list of implemented methods, transforms, and datasets.
For a quick start, check out our provided [examples](https://github.com/rusty1s/pytorch_geometric/tree/master/examples) in the `examples/` directory.

We are currently in our first alpha release and work on completing [documentation](http://rusty1s.github.io/pytorch_geometric).
If you notice anything unexpected, please open an [issue](https://github.com/rusty1s/pytorch_geometric/issues) and let us know.
If you are missing a specific method, feel free to open a [feature request](https://github.com/rusty1s/pytorch_geometric/issues).
We are constantly encouraged to make PyTorch Geometric even better.

## Installation

Ensure that at least PyTorch 0.4.1 is installed and verify that `cuda/bin` and `cuda/include` are in your `$PATH` and `$CPATH` respectively, *e.g.*:

```
$ python -c "import torch; print(torch.__version__)"
>>> 0.4.1

$ echo $PATH
>>> /usr/local/cuda/bin:...

$ echo $CPATH
>>> /usr/local/cuda/include:...
```

Then run:

```
$ pip install --upgrade torch-scatter
$ pip install --upgrade torch-sparse
$ pip install --upgrade torch-cluster
$ pip install --upgrade torch-spline-conv
$ pip install torch-geometric
```

If you are running into any installation problems, please create an [issue](https://github.com/rusty1s/pytorch_geometric/issues).

## Running examples

```
cd examples
python cora.py
```

## Cite

Please cite our paper (and the respective papers of the methods used) if you use this code in your own work:

```
@inproceedings{Fey/etal/2018,
  title={{SplineCNN}: Fast Geometric Deep Learning with Continuous {B}-Spline Kernels},
  author={Fey, Matthias and Lenssen, Jan Eric and Weichert, Frank and M{\"u}ller, Heinrich},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2018},
}
```

## Running tests

```
python setup.py test
```
