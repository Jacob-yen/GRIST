"""Tests for callbacks in training.py"""

from functools import partial
from unittest.mock import Mock
from unittest.mock import patch
from unittest.mock import call

import numpy as np
import pytest
from sklearn.base import clone


class TestCheckpoint:
    @pytest.fixture
    def checkpoint_cls(self):
        from scripts.study_case.ID_12.skorch.callbacks import Checkpoint
        return Checkpoint

    @pytest.fixture
    def save_params_mock(self):
        with patch('skorch.NeuralNet.save_params') as mock:
            yield mock

    @pytest.fixture
    def pickle_dump_mock(self):
        with patch('pickle.dump') as mock:
            yield mock

    @pytest.fixture
    def net_cls(self):
        """very simple network that trains for 10 epochs"""
        from scripts.study_case.ID_12.skorch import NeuralNetRegressor
        from scripts.study_case.ID_12.skorch.toy import make_regressor

        module_cls = make_regressor(
            input_units=1,
            num_hidden=0,
            output_units=1,
        )

        return partial(
            NeuralNetRegressor,
            module=module_cls,
            max_epochs=10,
            batch_size=10)

    @pytest.fixture(scope='module')
    def data(self):
        # have 10 examples so we can do a nice CV split
        X = np.zeros((10, 1), dtype='float32')
        y = np.zeros((10, 1), dtype='float32')
        return X, y

    def test_none_monitor_saves_always(
            self, save_params_mock, net_cls, checkpoint_cls, data):
        sink = Mock()
        net = net_cls(callbacks=[
            checkpoint_cls(monitor=None, sink=sink,
                           event_name='event_another'),
        ])
        net.fit(*data)

        assert save_params_mock.call_count == 3*len(net.history)
        assert sink.call_count == len(net.history)
        assert all((x is True) for x in net.history[:, 'event_another'])

    @pytest.mark.parametrize('message,files', [
        ('Unable to save model parameters to params.pt, '
         'Exception: encoding error',
         {'f_params': 'params.pt', 'f_optimizer': None, 'f_history': None}),
        ('Unable to save optimizer state to optimizer.pt, '
         'Exception: encoding error',
         {'f_params': None, 'f_optimizer': 'optimizer.pt', 'f_history': None}),
        ('Unable to save history to history.json, '
         'Exception: encoding error',
         {'f_params': None, 'f_optimizer': None, 'f_history': 'history.json'})
    ])
    def test_outputs_to_sink_when_save_params_errors(
            self, save_params_mock, net_cls, checkpoint_cls, data,
            message, files):
        sink = Mock()
        save_params_mock.side_effect = Exception('encoding error')
        net = net_cls(callbacks=[
            checkpoint_cls(monitor=None, sink=sink, **files)
        ])
        net.fit(*data)

        assert save_params_mock.call_count == len(net.history)
        assert sink.call_count == 2*len(net.history)
        save_error_messages = [call(message)] * len(net.history)
        sink.assert_has_calls(save_error_messages, any_order=True)

    @pytest.mark.parametrize('f_name, mode', [
        ('f_params', 'w'),
        ('f_optimizer', 'w'),
        ('f_history', 'w'),
        ('f_pickle', 'wb')
    ])
    def test_init_with_dirname_and_file_like_object_error(
            self, checkpoint_cls, tmpdir, f_name, mode):
        from scripts.study_case.ID_12.skorch.exceptions import SkorchException

        skorch_dir = tmpdir.mkdir("skorch")
        exp_dir = skorch_dir.join("exp1")
        f = skorch_dir.join(f_name + ".pt")

        with f.open(mode) as fp:
            with pytest.raises(SkorchException) as e:
                checkpoint_cls(**{f_name: fp}, dirname=str(exp_dir))
        expected = "dirname can only be used when f_* are strings"
        assert str(e.value) == expected

    @pytest.mark.parametrize('f_name, mode', [
        ('f_params', 'w'),
        ('f_optimizer', 'w'),
        ('f_history', 'w'),
        ('f_pickle', 'wb')
    ])
    def test_initialize_with_dirname_and_file_like_object_error(
            self, checkpoint_cls, tmpdir, f_name, mode):
        from scripts.study_case.ID_12.skorch.exceptions import SkorchException

        skorch_dir = tmpdir.mkdir("skorch")
        exp_dir = skorch_dir.join("exp1")
        f = skorch_dir.join(f_name + ".pt")

        with f.open(mode) as fp:
            with pytest.raises(SkorchException) as e:
                cp = checkpoint_cls(dirname=str(exp_dir))
                setattr(cp, f_name, fp)
                cp.initialize()
        expected = "dirname can only be used when f_* are strings"
        assert str(e.value) == expected

    def test_default_without_validation_raises_meaningful_error(
            self, net_cls, checkpoint_cls, data):
        net = net_cls(
            callbacks=[
                checkpoint_cls(),
            ],
            train_split=None
        )
        from scripts.study_case.ID_12.skorch.exceptions import SkorchException
        with pytest.raises(SkorchException) as e:
            net.fit(*data)
            expected = (
                "Monitor value '{}' cannot be found in history. "
                "Make sure you have validation data if you use "
                "validation scores for checkpointing.".format(
                    'valid_loss_best')
            )
            assert str(e.value) == expected

    def test_string_monitor_and_formatting(
            self, save_params_mock, net_cls, checkpoint_cls, data):
        def epoch_3_scorer(net, *_):
            return 1 if net.history[-1, 'epoch'] == 3 else 0

        from scripts.study_case.ID_12.skorch.callbacks import EpochScoring
        scoring = EpochScoring(
            scoring=epoch_3_scorer, on_train=True, lower_is_better=False)

        sink = Mock()
        cb = checkpoint_cls(
            monitor='epoch_3_scorer_best',
            f_params='model_{last_epoch[epoch]}_{net.max_epochs}.pt',
            f_optimizer='optimizer_{last_epoch[epoch]}_{net.max_epochs}.pt',
            sink=sink)
        net = net_cls(callbacks=[
            ('my_score', scoring), cb
        ])
        net.fit(*data)

        assert save_params_mock.call_count == 6
        assert cb.get_formatted_files(net) == {
            'f_params': 'model_3_10.pt',
            'f_optimizer': 'optimizer_3_10.pt',
            'f_history': 'history.json',
            'f_pickle': None
        }
        save_params_mock.assert_has_calls(
            [call(f_params='model_1_10.pt'),
             call(f_optimizer='optimizer_1_10.pt'),
             call(f_history='history.json'),
             call(f_params='model_3_10.pt'),
             call(f_optimizer='optimizer_3_10.pt'),
             call(f_history='history.json')]
        )
        assert sink.call_count == 2
        # The first epoch will always be saved. `epoch_3_scorer` returns 1 at
        # epoch 3, which will trigger another checkpoint. For all other epochs
        # `epoch_3_scorer` returns 0, which does not trigger a checkpoint.
        assert [True, False, True] + [False] * 7 == net.history[:, 'event_cp']

    def test_save_all_targets(
            self, save_params_mock, pickle_dump_mock,
            net_cls, checkpoint_cls, data):
        net = net_cls(callbacks=[
            checkpoint_cls(
                monitor=None, f_params='params.pt',
                f_history='history.json', f_pickle='model.pkl',
                f_optimizer='optimizer.pt'),
        ])
        net.fit(*data)

        assert save_params_mock.call_count == 3*len(net.history)
        assert pickle_dump_mock.call_count == len(net.history)

        print(save_params_mock.call_args_list)
        save_params_mock.assert_has_calls(
            [call(f_params='params.pt'),
             call(f_optimizer='optimizer.pt'),
             call(f_history='history.json')] * len(net.history)
        )

    def test_save_all_targets_with_prefix(
            self, save_params_mock, pickle_dump_mock,
            net_cls, checkpoint_cls, data):

        cp = checkpoint_cls(
            monitor=None,
            f_params='params.pt',
            f_history='history.json',
            f_pickle='model.pkl',
            f_optimizer='optimizer.pt',
            fn_prefix="exp1_")
        net = net_cls(callbacks=[cp])
        net.fit(*data)

        assert cp.f_history_ == "exp1_history.json"
        assert save_params_mock.call_count == 3*len(net.history)
        assert pickle_dump_mock.call_count == len(net.history)
        save_params_mock.assert_has_calls(
            [call(f_params='exp1_params.pt'),
             call(f_optimizer='exp1_optimizer.pt'),
             call(f_history='exp1_history.json')] * len(net.history)
        )

    def test_save_all_targets_with_prefix_and_dirname(
            self, save_params_mock, pickle_dump_mock,
            net_cls, checkpoint_cls, data, tmpdir):

        skorch_dir = tmpdir.mkdir('skorch').join('exp1')

        cp = checkpoint_cls(
            monitor=None,
            f_params='params.pt',
            f_history='history.json',
            f_pickle='model.pkl',
            f_optimizer='optimizer.pt',
            fn_prefix="unet_",
            dirname=str(skorch_dir))
        net = net_cls(callbacks=[cp])
        net.fit(*data)

        f_params = skorch_dir.join('unet_params.pt')
        f_optimizer = skorch_dir.join('unet_optimizer.pt')
        f_history = skorch_dir.join('unet_history.json')

        assert cp.f_history_ == str(f_history)
        assert save_params_mock.call_count == 3*len(net.history)
        assert pickle_dump_mock.call_count == len(net.history)
        save_params_mock.assert_has_calls(
            [call(f_params=str(f_params)),
             call(f_optimizer=str(f_optimizer)),
             call(f_history=str(f_history))] * len(net.history)
        )
        assert skorch_dir.exists()

    def test_save_no_targets(
            self, save_params_mock, pickle_dump_mock,
            net_cls, checkpoint_cls, data):
        net = net_cls(callbacks=[
            checkpoint_cls(
                monitor=None, f_params=None, f_optimizer=None,
                f_history=None, f_pickle=None),
        ])
        net.fit(*data)

        assert save_params_mock.call_count == 0
        assert pickle_dump_mock.call_count == 0

    def test_warnings_when_monitor_appears_in_history(
            self, net_cls, checkpoint_cls, save_params_mock, data):
        net = net_cls(callbacks=[
            checkpoint_cls(monitor="valid_loss")],
            max_epochs=1)

        exp_warn = (
            "Checkpoint monitor parameter is set to 'valid_loss' and the "
            "history contains 'valid_loss_best'. Perhaps you meant to set the "
            "parameter to 'valid_loss_best'")

        with pytest.warns(UserWarning, match=exp_warn):
            net.fit(*data)
        assert save_params_mock.call_count == 3


class TestEarlyStopping:

    @pytest.fixture
    def early_stopping_cls(self):
        from scripts.study_case.ID_12.skorch.callbacks import EarlyStopping
        return EarlyStopping

    @pytest.fixture
    def epoch_scoring_cls(self):
        from scripts.study_case.ID_12.skorch.callbacks import EpochScoring
        return EpochScoring

    @pytest.fixture
    def net_clf_cls(self):
        from scripts.study_case.ID_12.skorch import NeuralNetClassifier
        return NeuralNetClassifier

    @pytest.fixture
    def broken_classifier_module(self, classifier_module):
        """Return a classifier that does not improve over time."""
        class BrokenClassifier(classifier_module.func):
            def forward(self, x):
                return super().forward(x) * 0 + 0.5
        return BrokenClassifier

    def test_typical_use_case_nonstop(
            self, net_clf_cls, classifier_module, classifier_data,
            early_stopping_cls):
        patience = 5
        max_epochs = 8
        early_stopping_cb = early_stopping_cls(patience=patience)

        net = net_clf_cls(
            classifier_module,
            callbacks=[
                early_stopping_cb,
            ],
            max_epochs=max_epochs,
        )
        net.fit(*classifier_data)

        assert len(net.history) == max_epochs

    def test_typical_use_case_stopping(
            self, net_clf_cls, broken_classifier_module, classifier_data,
            early_stopping_cls):
        patience = 5
        max_epochs = 8
        side_effect = []

        def sink(x):
            side_effect.append(x)

        early_stopping_cb = early_stopping_cls(patience=patience, sink=sink)

        net = net_clf_cls(
            broken_classifier_module,
            callbacks=[
                early_stopping_cb,
            ],
            max_epochs=max_epochs,
        )
        net.fit(*classifier_data)

        assert len(net.history) == patience + 1 < max_epochs

        # check correct output message
        assert len(side_effect) == 1
        msg = side_effect[0]
        expected_msg = ("Stopping since valid_loss has not improved in "
                        "the last 5 epochs.")
        assert msg == expected_msg

    def test_custom_scoring_nonstop(
            self, net_clf_cls, classifier_module, classifier_data,
            early_stopping_cls, epoch_scoring_cls,
    ):
        lower_is_better = False
        scoring_name = 'valid_roc_auc'
        patience = 5
        max_epochs = 8
        scoring_mock = Mock(side_effect=list(range(2, 10)))
        scoring_cb = epoch_scoring_cls(
            scoring_mock, lower_is_better, name=scoring_name)
        early_stopping_cb = early_stopping_cls(
            patience=patience, lower_is_better=lower_is_better,
            monitor=scoring_name)

        net = net_clf_cls(
            classifier_module,
            callbacks=[
                scoring_cb,
                early_stopping_cb,
            ],
            max_epochs=max_epochs,
        )
        net.fit(*classifier_data)

        assert len(net.history) == max_epochs

    def test_custom_scoring_stop(
            self, net_clf_cls, broken_classifier_module, classifier_data,
            early_stopping_cls, epoch_scoring_cls,
    ):
        lower_is_better = False
        scoring_name = 'valid_roc_auc'
        patience = 5
        max_epochs = 8
        scoring_cb = epoch_scoring_cls(
            'roc_auc', lower_is_better, name=scoring_name)
        early_stopping_cb = early_stopping_cls(
            patience=patience, lower_is_better=lower_is_better,
            monitor=scoring_name)

        net = net_clf_cls(
            broken_classifier_module,
            callbacks=[
                scoring_cb,
                early_stopping_cb,
            ],
            max_epochs=max_epochs,
        )
        net.fit(*classifier_data)

        assert len(net.history) < max_epochs

    def test_stopping_big_absolute_threshold(
            self, net_clf_cls, classifier_module, classifier_data,
            early_stopping_cls):
        patience = 5
        max_epochs = 8
        early_stopping_cb = early_stopping_cls(patience=patience,
                                               threshold_mode='abs',
                                               threshold=0.1)

        net = net_clf_cls(
            classifier_module,
            callbacks=[
                early_stopping_cb,
            ],
            max_epochs=max_epochs,
        )
        net.fit(*classifier_data)

        assert len(net.history) == patience + 1 < max_epochs

    def test_wrong_threshold_mode(
            self, net_clf_cls, classifier_module, classifier_data,
            early_stopping_cls):
        patience = 5
        max_epochs = 8
        early_stopping_cb = early_stopping_cls(
            patience=patience, threshold_mode='incorrect')
        net = net_clf_cls(
            classifier_module,
            callbacks=[
                early_stopping_cb,
            ],
            max_epochs=max_epochs,
        )

        with pytest.raises(ValueError) as exc:
            net.fit(*classifier_data)

        expected_msg = "Invalid threshold mode: 'incorrect'"
        assert exc.value.args[0] == expected_msg



class TestParamMapper:

    @pytest.fixture
    def initializer(self):
        from scripts.study_case.ID_12.skorch.callbacks import Initializer
        return Initializer

    @pytest.fixture
    def freezer(self):
        from scripts.study_case.ID_12.skorch.callbacks import Freezer
        return Freezer

    @pytest.fixture
    def unfreezer(self):
        from scripts.study_case.ID_12.skorch.callbacks import Unfreezer
        return Unfreezer

    @pytest.fixture
    def param_mapper(self):
        from scripts.study_case.ID_12.skorch.callbacks import ParamMapper
        return ParamMapper

    @pytest.fixture
    def net_cls(self):
        from scripts.study_case.ID_12.skorch import NeuralNetClassifier
        return NeuralNetClassifier

    @pytest.mark.parametrize('at', [0, -1])
    def test_subzero_at_fails(self, net_cls, classifier_module,
                              param_mapper, at):
        cb = param_mapper(patterns='*', at=at)
        net = net_cls(classifier_module, callbacks=[cb])
        with pytest.raises(ValueError):
            net.initialize()

    @pytest.mark.parametrize('mod_init', [False, True])
    @pytest.mark.parametrize('weight_pattern', [
        'sequential.*.weight',
        lambda name: name.startswith('sequential') and name.endswith('.weight'),
    ])
    def test_initialization_is_effective(self, net_cls, classifier_module,
                                         classifier_data, initializer,
                                         mod_init, weight_pattern):
        from torch.nn.init import constant_
        from scripts.study_case.ID_12.skorch.utils import to_numpy

        module = classifier_module() if mod_init else classifier_module

        net = net_cls(
            module,
            lr=0,
            max_epochs=1,
            callbacks=[
                initializer(weight_pattern, partial(constant_, val=5)),
                initializer('sequential.3.bias', partial(constant_, val=10)),
            ])

        net.fit(*classifier_data)

        assert np.allclose(to_numpy(net.module_.sequential[0].weight), 5)
        assert np.allclose(to_numpy(net.module_.sequential[3].weight), 5)
        assert np.allclose(to_numpy(net.module_.sequential[3].bias), 10)

    @pytest.mark.parametrize('mod_init', [False, True])
    @pytest.mark.parametrize('mod_kwargs', [
        {},
        # Supply a module__ parameter so the model is forced
        # to re-initialize. Even then parameters should be
        # frozen correctly.
        {'module__hidden_units': 5},
    ])
    def test_freezing_is_effective(self, net_cls, classifier_module,
                                   classifier_data, freezer, mod_init,
                                   mod_kwargs):
        from scripts.study_case.ID_12.skorch.utils import to_numpy

        module = classifier_module() if mod_init else classifier_module

        net = net_cls(
            module,
            max_epochs=2,
            callbacks=[
                freezer('sequential.*.weight'),
                freezer('sequential.3.bias'),
            ],
            **mod_kwargs)

        net.initialize()

        assert net.module_.sequential[0].weight.requires_grad
        assert net.module_.sequential[3].weight.requires_grad
        assert net.module_.sequential[0].bias.requires_grad
        assert net.module_.sequential[3].bias.requires_grad

        dense0_weight_pre = to_numpy(net.module_.sequential[0].weight).copy()
        dense1_weight_pre = to_numpy(net.module_.sequential[3].weight).copy()
        dense0_bias_pre = to_numpy(net.module_.sequential[0].bias).copy()
        dense1_bias_pre = to_numpy(net.module_.sequential[3].bias).copy()

        # use partial_fit to not re-initialize the module (weights)
        net.partial_fit(*classifier_data)

        dense0_weight_post = to_numpy(net.module_.sequential[0].weight).copy()
        dense1_weight_post = to_numpy(net.module_.sequential[3].weight).copy()
        dense0_bias_post = to_numpy(net.module_.sequential[0].bias).copy()
        dense1_bias_post = to_numpy(net.module_.sequential[3].bias).copy()

        assert not net.module_.sequential[0].weight.requires_grad
        assert not net.module_.sequential[3].weight.requires_grad
        assert net.module_.sequential[0].bias.requires_grad
        assert not net.module_.sequential[3].bias.requires_grad

        assert np.allclose(dense0_weight_pre, dense0_weight_post)
        assert np.allclose(dense1_weight_pre, dense1_weight_post)
        assert not np.allclose(dense0_bias_pre, dense0_bias_post)
        assert np.allclose(dense1_bias_pre, dense1_bias_post)

    def test_unfreezing_is_effective(self, net_cls, classifier_module,
                                     classifier_data, freezer, unfreezer):
        from scripts.study_case.ID_12.skorch.utils import to_numpy

        net = net_cls(
            classifier_module,
            max_epochs=1,
            callbacks=[
                freezer('sequential.*.weight'),
                freezer('sequential.3.bias'),
                unfreezer('sequential.*.weight', at=2),
                unfreezer('sequential.3.bias', at=2),
            ])

        net.initialize()

        # epoch 1, freezing parameters
        net.partial_fit(*classifier_data)

        assert not net.module_.sequential[0].weight.requires_grad
        assert not net.module_.sequential[3].weight.requires_grad
        assert net.module_.sequential[0].bias.requires_grad
        assert not net.module_.sequential[3].bias.requires_grad

        dense0_weight_pre = to_numpy(net.module_.sequential[0].weight).copy()
        dense1_weight_pre = to_numpy(net.module_.sequential[3].weight).copy()
        dense0_bias_pre = to_numpy(net.module_.sequential[0].bias).copy()
        dense1_bias_pre = to_numpy(net.module_.sequential[3].bias).copy()

        # epoch 2, unfreezing parameters
        net.partial_fit(*classifier_data)

        assert net.module_.sequential[0].weight.requires_grad
        assert net.module_.sequential[3].weight.requires_grad
        assert net.module_.sequential[0].bias.requires_grad
        assert net.module_.sequential[3].bias.requires_grad

        # epoch 3, modifications should have been made
        net.partial_fit(*classifier_data)

        dense0_weight_post = to_numpy(net.module_.sequential[0].weight).copy()
        dense1_weight_post = to_numpy(net.module_.sequential[3].weight).copy()
        dense0_bias_post = to_numpy(net.module_.sequential[0].bias).copy()
        dense1_bias_post = to_numpy(net.module_.sequential[3].bias).copy()

        assert not np.allclose(dense0_weight_pre, dense0_weight_post)
        assert not np.allclose(dense1_weight_pre, dense1_weight_post)
        assert not np.allclose(dense0_bias_pre, dense0_bias_post)
        assert not np.allclose(dense1_bias_pre, dense1_bias_post)


    def test_schedule_is_effective(self, net_cls, classifier_module,
                                   classifier_data, param_mapper):
        from scripts.study_case.ID_12.skorch.utils import to_numpy, noop
        from scripts.study_case.ID_12.skorch.utils import freeze_parameter, unfreeze_parameter

        def schedule(net):
            if len(net.history) == 1:
                return freeze_parameter
            elif len(net.history) == 2:
                return unfreeze_parameter
            return noop

        net = net_cls(
            classifier_module,
            max_epochs=1,
            callbacks=[
                param_mapper(
                    ['sequential.*.weight', 'sequential.3.bias'],
                    schedule=schedule,
                ),
            ])

        net.initialize()

        # epoch 1, freezing parameters
        net.partial_fit(*classifier_data)

        assert not net.module_.sequential[0].weight.requires_grad
        assert not net.module_.sequential[3].weight.requires_grad
        assert net.module_.sequential[0].bias.requires_grad
        assert not net.module_.sequential[3].bias.requires_grad

        dense0_weight_pre = to_numpy(net.module_.sequential[0].weight).copy()
        dense1_weight_pre = to_numpy(net.module_.sequential[3].weight).copy()
        dense0_bias_pre = to_numpy(net.module_.sequential[0].bias).copy()
        dense1_bias_pre = to_numpy(net.module_.sequential[3].bias).copy()

        # epoch 2, unfreezing parameters
        net.partial_fit(*classifier_data)

        assert net.module_.sequential[0].weight.requires_grad
        assert net.module_.sequential[3].weight.requires_grad
        assert net.module_.sequential[0].bias.requires_grad
        assert net.module_.sequential[3].bias.requires_grad

        # epoch 3, modifications should have been made
        net.partial_fit(*classifier_data)

        dense0_weight_post = to_numpy(net.module_.sequential[0].weight).copy()
        dense1_weight_post = to_numpy(net.module_.sequential[3].weight).copy()
        dense0_bias_post = to_numpy(net.module_.sequential[0].bias).copy()
        dense1_bias_post = to_numpy(net.module_.sequential[3].bias).copy()

        assert not np.allclose(dense0_weight_pre, dense0_weight_post)
        assert not np.allclose(dense1_weight_pre, dense1_weight_post)
        assert not np.allclose(dense0_bias_pre, dense0_bias_post)
        assert not np.allclose(dense1_bias_pre, dense1_bias_post)


class TestLoadInitState:

    @pytest.fixture
    def checkpoint_cls(self):
        from scripts.study_case.ID_12.skorch.callbacks import Checkpoint
        return Checkpoint

    @pytest.fixture
    def loadinitstate_cls(self):
        from scripts.study_case.ID_12.skorch.callbacks import LoadInitState
        return LoadInitState

    @pytest.fixture
    def net_cls(self):
        """very simple network that trains for 10 epochs"""
        from scripts.study_case.ID_12.skorch import NeuralNetRegressor
        from scripts.study_case.ID_12.skorch.toy import make_regressor

        module_cls = make_regressor(
            input_units=1,
            num_hidden=0,
            output_units=1,
        )

        return partial(
            NeuralNetRegressor,
            module=module_cls,
            max_epochs=10,
            batch_size=10)

    @pytest.fixture(scope='module')
    def data(self):
        # have 10 examples so we can do a nice CV split
        X = np.zeros((10, 1), dtype='float32')
        y = np.zeros((10, 1), dtype='float32')
        return X, y

    def test_load_initial_state(
            self, checkpoint_cls, net_cls, loadinitstate_cls,
            data, tmpdir):
        skorch_dir = tmpdir.mkdir('skorch')
        f_params = skorch_dir.join('params.pt')
        f_optimizer = skorch_dir.join('optimizer.pt')
        f_history = skorch_dir.join('history.json')

        cp = checkpoint_cls(
            monitor=None,
            f_params=str(f_params),
            f_optimizer=str(f_optimizer),
            f_history=str(f_history)
        )
        load_init_state = loadinitstate_cls(cp)
        net = net_cls(callbacks=[cp, load_init_state])
        net.fit(*data)

        assert f_params.exists()
        assert f_optimizer.exists()
        assert f_history.exists()

        assert len(net.history) == 10
        del net

        new_net = net_cls(callbacks=[cp, load_init_state])
        new_net.fit(*data)

        assert len(new_net.history) == 20

    def test_load_initial_state_custom_scoring(
            self, checkpoint_cls, net_cls, loadinitstate_cls,
            data, tmpdir):
        def epoch_3_scorer(net, *_):
            return 1 if net.history[-1, 'epoch'] == 3 else 0

        from scripts.study_case.ID_12.skorch.callbacks import EpochScoring
        scoring = EpochScoring(
            scoring=epoch_3_scorer, on_train=True, lower_is_better=False)

        skorch_dir = tmpdir.mkdir('skorch')
        f_params = skorch_dir.join(
            'model_epoch_{last_epoch[epoch]}.pt')
        f_optimizer = skorch_dir.join(
            'optimizer_epoch_{last_epoch[epoch]}.pt')
        f_history = skorch_dir.join(
            'history.json')

        cp = checkpoint_cls(
            monitor='epoch_3_scorer_best',
            f_params=str(f_params),
            f_optimizer=str(f_optimizer),
            f_history=str(f_history)
        )
        load_init_state = loadinitstate_cls(cp)
        net = net_cls(callbacks=[load_init_state, scoring, cp])

        net.fit(*data)

        assert skorch_dir.join('model_epoch_3.pt').exists()
        assert skorch_dir.join('optimizer_epoch_3.pt').exists()
        assert skorch_dir.join('history.json').exists()

        assert len(net.history) == 10
        del net

        new_net = net_cls(callbacks=[load_init_state, scoring, cp])
        new_net.fit(*data)

        # new_net starts from the best epoch of the first run
        # the best epcoh of the previous run was at epoch 3
        # the second run went through 10 epochs, thus
        # 3 + 10 = 13
        assert len(new_net.history) == 13
        assert new_net.history[:, 'event_cp'] == [
            True, False, True] + [False] * 10


class TestTrainEndCheckpoint:
    @pytest.fixture
    def finalcheckpoint_cls(self):
        from scripts.study_case.ID_12.skorch.callbacks import TrainEndCheckpoint
        return TrainEndCheckpoint

    @pytest.fixture
    def save_params_mock(self):
        with patch('skorch.NeuralNet.save_params') as mock:
            yield mock

    @pytest.fixture
    def pickle_dump_mock(self):
        with patch('pickle.dump') as mock:
            yield mock

    @pytest.fixture
    def net_cls(self):
        """very simple network that trains for 10 epochs"""
        from scripts.study_case.ID_12.skorch import NeuralNetRegressor
        from scripts.study_case.ID_12.skorch.toy import make_regressor

        module_cls = make_regressor(
            input_units=1,
            num_hidden=0,
            output_units=1,
        )

        return partial(
            NeuralNetRegressor,
            module=module_cls,
            max_epochs=10,
            batch_size=10)

    @pytest.fixture(scope='module')
    def data(self):
        # have 10 examples so we can do a nice CV split
        X = np.zeros((10, 1), dtype='float32')
        y = np.zeros((10, 1), dtype='float32')
        return X, y

    def test_saves_at_end(
            self, save_params_mock, net_cls, finalcheckpoint_cls, data):
        sink = Mock()
        net = net_cls(callbacks=[
            finalcheckpoint_cls(
                sink=sink, dirname='exp1', fn_prefix='train_end_')
        ])
        net.fit(*data)

        assert save_params_mock.call_count == 3
        assert sink.call_args == call("Final checkpoint triggered")
        save_params_mock.assert_has_calls([
            call(f_params='exp1/train_end_params.pt'),
            call(f_optimizer='exp1/train_end_optimizer.pt'),
            call(f_history='exp1/train_end_history.json')
        ])

    def test_saves_at_end_with_custom_formatting(
            self, save_params_mock, net_cls, finalcheckpoint_cls, data):
        sink = Mock()
        net = net_cls(callbacks=[
            finalcheckpoint_cls(
                sink=sink, dirname='exp1',
                f_params='model_{last_epoch[epoch]}.pt',
                f_optimizer='optimizer_{last_epoch[epoch]}.pt',
                fn_prefix='train_end_'
            )
        ])
        net.fit(*data)

        assert save_params_mock.call_count == 3
        assert sink.call_args == call("Final checkpoint triggered")
        save_params_mock.assert_has_calls([
            call(f_params='exp1/train_end_model_10.pt'),
            call(f_optimizer='exp1/train_end_optimizer_10.pt'),
            call(f_history='exp1/train_end_history.json')
        ])

    def test_cloneable(self, finalcheckpoint_cls):
        # reproduces bug #459
        cp = finalcheckpoint_cls()
        clone(cp)  # does not raise
