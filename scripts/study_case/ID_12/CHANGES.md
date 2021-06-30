# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

### Changed

### Fixed


## [0.6.0] - 2019-07-19

### Added

- Adds FAQ entry regarding the initialization behavior of `NeuralNet` when passed instantiated models. (#409)
- Added CUDA pickle test including an artifact that supports testing on CUDA-less CI machines
- Adds `train_batch_count` and `valid_batch_count` to history in training loop. (#445)
- Adds score method for NeuralNetClassifier, NeuralNetBinaryClassifier, and NeuralNetRegressor (#469)
- Wrapper class for torch Datasets to make them work with some sklearn features (e.g. grid search). (#443)

### Changed

- Repository moved to https://github.com/skorch-dev/skorch/, please change your git remotes
- Treat cuda dependent attributes as prefix to cover values set using `set_params` since
  previously `"criterion_"` would not match `net.criterion__weight` as set by
  `net.set_params(criterion__weight=w)`
- skorch pickle format changed in order to improve CUDA compatibility, if you have pickled models, please re-pickle them to be able to load them in the future
- `net.criterion_` and its parameters are now moved to target device when using criteria that inherit from `torch.nn.Module`. Previously the user had to make sure that parameters such as class weight are on the compute device
- skorch now assumes PyTorch >= 1.1.0. This mainly affects learning rate schedulers, whose inner workings have been changed with version 1.1.0. This update will also invalidate pickled skorch models after a change introduced in PyTorch optimizers.

### Fixed

- Include requirements in MANIFEST.in
- Add `criterion_` to `NeuralNet.cuda_dependent_attributes_` to avoid issues with criterion
  weight tensors from, e.g., `NLLLoss` (#426)
- `TrainEndCheckpoint` can be cloned by `sklearn.base.clone`. (#459)


## [0.5.0] - 2018-12-13

### Added

- [Basic usage notebook][1810251445] now runs on Google Colab
- [Advanced usage notebook][1810261633] now runs on Google Colab
- [MNIST with scikit-learn and skorch][1811011230] now runs on Google Colab
- Better user-facing messages when module or optimizer are re-initialized
- Added an experimental API (`net._register_virtual_param`) to register "virtual"
  parameters on the network with custom setter functions. (#369)
- Setting parameters `lr`, `momentum`, `optimizer__lr`, etc. no longer resets
  the optmizer. As of now you can do `net.set_params(lr=0.03)` or
  `net.set_params(optimizer__param_group__0__momentum=0.86)` without triggering
  a re-initialization of the optimizer (#369)
- Support for scipy sparse CSR matrices as input (as, e.g., returned by sklearn's
  `CountVectorizer`); note that they are cast to dense matrices during batching
- Helper functions to build command line interfaces with almost no
  boilerplate, [example][1811191713] that shows usage

[1810251445]: https://colab.research.google.com/github/dnouri/skorch/blob/master/notebooks/Basic_Usage.ipynb
[1810261633]: https://colab.research.google.com/github/dnouri/skorch/blob/master/notebooks/Advanced_Usage.ipynb
[1811011230]: https://colab.research.google.com/github/dnouri/skorch/blob/master/notebooks/MNIST.ipynb
[1811191713]: https://github.com/dnouri/skorch/tree/master/examples/cli

### Changed

- Reduce overhead of `BatchScoring` when using `train_loss_score` or `valid_loss_score` by skipping superfluous inference step (#381)
- The `on_grad_computed` callback function will yield an iterable for `named_parameters` only when it is used to reduce the run-time overhead of the call (#379)
- Default `fn_prefix` in `TrainEndCheckpoint` is now `train_end_` (#391)
- Issues a warning when `Checkpoints`'s `monitor` parameter is set to `monitor` and the history contains `<monitor>_best`. (#399)

### Fixed

- Re-initialize optimizer when `set_params` is called with `lr` argument (#372)
- Copying a `SliceDict` now returns a `SliceDict` instead of a `dict` (#388)
- Calling `==` on `SliceDict`s now works as expected when values are numpy arrays and torch tensors


## [0.4.0] - 2018-10-24

### Added

- Support for PyTorch 0.4.1
- There is no need to explicitly name callbacks anymore (names are assigned automatically, name conflicts are resolved).
- You can now access the training data in the `on_grad_computed` event
- There is a new [image segmentation example][1]
- Easily create toy network instances for quick experiments using [`skorch.toy.make_classifier`][2] and friends
- New [`ParamMapper`][3] callback to modify/freeze/unfreeze parameters at certain point in time during training:
```python
>>> from sklearn.callbacks import Freezer, Unfreezer
>>> net = Net(module, callbacks=[Freezer('layer*.weight'), Unfreezer('layer*.weight', at=10)])
```
- Refactored `EpochScoring` for easier sub-classing
- `Checkpoint` callback now supports saving the optimizer, this avoids problems with stateful
  optimizers such as `Adam` or `RMSprop` (#360)
- Added `LoadInitState` callback for easy continued training from checkpoints (#360)
- `NeuralNetwork.load_params` now supports loading from `Checkpoint` instances
- Added documentation for [saving and loading][4]

[1]: https://nbviewer.jupyter.org/github/dnouri/skorch/blob/master/examples/nuclei_image_segmentation/Nuclei_Image_Segmentation.ipynb
[2]: https://skorch.readthedocs.io/en/latest/toy.html
[3]: https://skorch.readthedocs.io/en/latest/callbacks.html#skorch.callbacks.ParamMapper
[4]: https://skorch.readthedocs.io/en/latest/user/save_load.html

### Changed

- The `ProgressBar` callback now determines the batches per epoch automatically by default (`batches_per_epoch=auto`)
- The `on_grad_computed` event now has access to the current training data batch

### Deprecated

- Deprecated `filtered_optimizer` in favor of `Freezer` callback (#346)
- `NeuralNet.load_params` and `NeuralNet.save_params` deprecate `f` parameter for the sake
  of `f_optimizer`, `f_params` and `f_history` (#360)

### Fixed

- `uses_placeholder_y` should not require existence of `y` field (#311)
- LR scheduler creates `batch_idx` on first run (#314)
- Use `OrderedDict` for callbacks to fix python 3.5 compatibility issues (#331)
- Make `to_tensor` work correctly with `PackedSequence` (#335)
- Rewrite `History` to not use any recursion to avoid memory leaks during exceptions (#312)
- Use `flaky` in some neural network tests to hide platform differences
- Fixes ReduceLROnPlateau when mode == max (#363)
- Fix disconnected weights between net and optimizer after copying the net with `copy.deepcopy` (#318)
- Fix a bug that intefered with loading CUDA models when the model was a CUDA tensor but
  the net was configured to use the CPU (#354, #358)


[Unreleased]: https://github.com/dnouri/skorch/compare/v0.5.0...HEAD
[0.4.0]: https://github.com/dnouri/skorch/compare/v0.3.0...v0.4.0
[0.5.0]: https://github.com/dnouri/skorch/compare/v0.4.0...v0.5.0
