skorch documentation
====================

A scikit-learn compatible neural network library that wraps PyTorch.


Introduction
------------

The goal of skorch is to make it possible to use PyTorch_ with
sklearn_. This is achieved by providing a wrapper around
PyTorch that has an sklearn interface. In that sense,
skorch is the spiritual successor to nolearn_, but instead of
using Lasagne and Theano, it uses PyTorch.

skorch does not re-invent the wheel, instead getting as much out
of your way as possible. If you are familiar with sklearn and
PyTorch, you don't have to learn any new concepts, and the syntax
should be well known. (If you're not familiar with those libraries, it
is worth getting familiarized.)

Additionally, skorch abstracts away the training loop, making a
lot of boilerplate code obsolete. A simple ``net.fit(X, y)`` is
enough. Out of the box, skorch works with many types of data, be
it PyTorch Tensors, NumPy arrays, Python dicts, and so
on. However, if you have other data, extending skorch is easy to
allow for that.

Overall, skorch aims at being as flexible as PyTorch while
having a clean interface as sklearn.


User's Guide
------------
.. toctree::
   :maxdepth: 2

   user/installation
   user/quickstart
   user/tutorials
   user/neuralnet
   user/callbacks
   user/dataset
   user/save_load
   user/history
   user/toy
   user/helper
   user/REST
   user/parallelism
   user/FAQ


API Reference
-------------

If you are looking for information on a specific function, class or
method, this part of the documentation is for you.

.. toctree::
  :maxdepth: 2

  skorch API <skorch>


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


.. _nolearn: https://github.com/dnouri/nolearn
.. _pytorch: http://pytorch.org/
.. _sklearn: http://scikit-learn.org/
