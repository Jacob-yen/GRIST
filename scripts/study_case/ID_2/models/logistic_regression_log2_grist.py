import tensorflow as tf
import numpy as np
import sys
sys.path.append("/data")
from scripts.study_case.ID_2.utils import utilities
import scripts.study_case.ID_2.model_log2 as model
import math



class LogisticRegression(model.Model):
    """Simple Logistic Regression using TensorFlow.
    The interface of the class is sklearn-like.
    """

    def __init__(self, main_dir='lr/', model_name='lr', loss_func='cross_entropy', dataset='mnist',
                 learning_rate=0.01, verbose=0, num_epochs=10, batch_size=10, start_time=None):

        """
        :param verbose: Level of verbosity. 0 - silent, 1 - print accuracy.
        """
        model.Model.__init__(self, model_name, main_dir)

        self._initialize_training_parameters(loss_func, learning_rate, num_epochs, batch_size,
                                             dataset, None, None)

        self.verbose = verbose

        # Computational graph nodes
        self.input_data = None
        self.input_labels = None

        self.W_ = None
        self.b_ = None

        self.model_output = None
        self.start_time = start_time
        self.accuracy = None
        self.obj_grad = None
        self.accuracy = None
        self.start_time = start_time
        self.safe_ptr = 0

        self.bottom_k = int(batch_size * 0.05)
        self.clip_cnt = [0 for _ in range(batch_size)]
        self.update_cnt = [0 for _ in range(batch_size)]
        self.iter_cnt = [0 for _ in range(batch_size)]
        self.useful_update_cnt = [0 for _ in range(batch_size)]

    def build_model(self, n_features, n_classes):

        """ Creates the computational graph.
        :param n_features: number of features
        :param n_classes: number of classes
        :return: self
        """

        self._create_placeholders(n_features, n_classes)
        self._create_variables(n_features, n_classes)

        self.model_output = tf.nn.softmax(tf.matmul(self.input_data, self.W_) + self.b_)

        self._create_cost_function_node(self.loss_func, self.model_output, self.input_labels)
        self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)
        self._create_test_node()

        self.obj_grad = tf.gradients(self.obj_function, self.input_data)[0]

    def _create_placeholders(self, n_features, n_classes):

        """ Create the TensorFlow placeholders for the model.
        :param n_features: number of features
        :param n_classes: number of classes
        :return: self
        """

        self.input_data = tf.placeholder("float", [None, n_features], name='x-input')
        self.input_labels = tf.placeholder("float", [None, n_classes], name='y-input')

    def _create_variables(self, n_features, n_classes):

        """ Create the TensorFlow variables for the model.
        :param n_features: number of features
        :param n_classes: number of classes
        :return: self
        """

        self.W_ = tf.Variable(tf.zeros([n_features, n_classes]), name='weights')
        self.b_ = tf.Variable(tf.zeros([n_classes]), name='biases')

    def _create_test_node(self):

        """
        :return:
        """

        with tf.name_scope("test"):
            correct_prediction = tf.equal(tf.argmax(self.model_output, 1), tf.argmax(self.input_labels, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            _ = tf.summary.scalar('accuracy', self.accuracy)

    def fit(self, train_set, train_labels, validation_set=None, validation_labels=None, restore_previous_model=False):

        """ Fit the model to the data.
        :param train_set: Training data. shape(n_samples, n_features).
        :param train_labels: Labels for the data. shape(n_samples, n_classes).
        :param validation_set: optional, default None. Validation data. shape(n_validation_samples, n_features).
        :param validation_labels: optional, default None. Labels for the validation data. shape(n_validation_samples, n_classes).
        :param restore_previous_model:
                    if true, a previous trained model
                    with the same name of this model is restored from disk to continue training.
        :return: self
        """

        with tf.Session() as self.tf_session:
            self._initialize_tf_utilities_and_ops(restore_previous_model)
            self._train_model(train_set, train_labels, validation_set, validation_labels)
            self.tf_saver.save(self.tf_session, self.models_dir + self.model_name)

    def _train_model(self, train_set, train_labels, validation_set, validation_labels):

        """ Train the model.
        :param train_set: training set
        :param train_labels: training labels
        :param validation_set: validation set
        :param validation_labels: validation labels
        :return: self
        """
        # from tensorflow.python import debug as tf_debug
        # self.tf_session = tf_debug.LocalCLIDebugWrapperSession(self.tf_session)
        # self.tf_session.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

        # for i in range(self.num_epochs):


        """insert code"""
        from scripts.utils.tf_utils import GradientSearcher
        gradient_search = GradientSearcher(name="ghips6_log2_grist")

        shuff = list(zip(train_set, train_labels))
        np.random.shuffle(shuff)
        batches = [_ for _ in utilities.gen_batches(shuff, self.batch_size)]
        batch = batches[0]
        batch_xs, batch_ys = zip(*batch)
        batch_xs = np.array(list(batch_xs))
        batch_ys = np.array(list(batch_ys))

        max_val, min_val = np.max(batch_xs), np.min(batch_xs)
        gradient_search.build(batch_size=self.batch_size, min_val=min_val, max_val=max_val)
        """insert code"""

        while True:

            """inserted code"""
            monitor_vars = {'loss': self.cost, 'obj_function': self.obj_function, 'obj_grad': self.obj_grad}
            feed_dict = {self.input_data: batch_xs, self.input_labels: batch_ys}
            batch_xs, scores_rank = gradient_search.update_batch_data(session=self.tf_session, monitor_var=monitor_vars,
                                                                      feed_dict=feed_dict, input_data=batch_xs,
                                                                      )
            """inserted code"""

            self.tf_session.run(self.train_step, feed_dict={self.input_data: batch_xs, self.input_labels: batch_ys})

            """inserted code"""
            total_batch = math.floor(len(batches)/self.batch_size)
            batch_idx = gradient_search.iter_count % total_batch
            new_batch = batches[batch_idx]
            new_batch_xs, new_batch_ys = zip(*new_batch)
            new_batch_xs = np.array(list(new_batch_xs))
            new_batch_ys = np.array(list(new_batch_ys))
            new_data_dict = {'x': new_batch_xs, 'y': new_batch_ys}
            old_data_dict = {'x': batch_xs, 'y': batch_ys}
            batch_xs, batch_ys = gradient_search.switch_new_data(new_data_dict=new_data_dict,
                                                                 old_data_dict=old_data_dict,
                                                                 scores_rank=scores_rank)
            gradient_search.check_time()
            """inserted code"""

    def _run_validation_error_and_summaries(self, epoch, validation_set, validation_labels):

        """ Run the summaries and error computation on the validation set.
        :param epoch: current epoch
        :param validation_set: validation set
        :param validation_labels: validation labels
        :return: self
        """

        feed = {self.input_data: validation_set, self.input_labels: validation_labels}
        result = self.tf_session.run([self.tf_merged_summaries, self.accuracy], feed_dict=feed)
        summary_str = result[0]
        acc = result[1]

        self.tf_summary_writer.add_summary(summary_str, epoch)

        if self.verbose == 1:
            print("Accuracy at step %s: %s" % (epoch, acc))

    def predict(self, test_set, test_labels):

        """ Compute the accuracy over the test set.
        :param test_set: Testing data. shape(n_test_samples, n_features).
        :param test_labels: Labels for the test data. shape(n_test_samples, n_classes).
        :return: accuracy
        """

        with tf.Session() as self.tf_session:
            self.tf_saver.restore(self.tf_session, self.models_dir + self.model_name)
            return self.accuracy.eval({self.input_data: test_set, self.input_labels: test_labels})


# import tensorflow as tf
# import numpy as np
# import sys
# from scripts.study_case.GH_IPS6.utils import utilities
# import scripts.study_case.GH_IPS6.model as model
# from datetime import datetime
#
# """inserted code"""
# from scripts.utils.utils import Utils
# s1 = datetime.now()
# score_mthd = 1
# drop_rate = 0.05
# # update_stragety = 'useful'
# update_stragety = 'rate'
# if score_mthd == 1:
#     get_score = Utils.get_score1
# elif score_mthd == 2:
#     get_score = Utils.get_score2
# else:
#     get_score = Utils.get_score3
#
# if update_stragety == 'useful':
#     get_score = Utils.get_score3
# print(get_score)
# """inserted code"""
#
# class LogisticRegression(model.Model):
#     """Simple Logistic Regression using TensorFlow.
#     The interface of the class is sklearn-like.
#     """
#
#     def __init__(self, main_dir='lr/', model_name='lr', loss_func='cross_entropy', dataset='mnist',
#                  learning_rate=0.01, verbose=0, num_epochs=10, batch_size=10,start_time=None):
#
#         """
#         :param verbose: Level of verbosity. 0 - silent, 1 - print accuracy.
#         """
#         model.Model.__init__(self, model_name, main_dir)
#
#         self._initialize_training_parameters(loss_func, learning_rate, num_epochs, batch_size,
#                                              dataset, None, None)
#
#         self.verbose = verbose
#
#         # Computational graph nodes
#         self.input_data = None
#         self.input_labels = None
#
#         self.W_ = None
#         self.b_ = None
#
#         self.model_output = None
#         self.start_time = start_time
#         self.accuracy = None
#         self.obj_grad = None
#         self.accuracy = None
#         self.start_time = start_time
#         self.safe_ptr = 0
#
#         self.bottom_k = int(batch_size * 0.05)
#         self.clip_cnt = [0 for _ in range(batch_size)]
#         self.update_cnt = [0 for _ in range(batch_size)]
#         self.iter_cnt = [0 for _ in range(batch_size)]
#         self.useful_update_cnt = [0 for _ in range(batch_size)]
#
#
#     def build_model(self, n_features, n_classes):
#
#         """ Creates the computational graph.
#         :param n_features: number of features
#         :param n_classes: number of classes
#         :return: self
#         """
#
#         self._create_placeholders(n_features, n_classes)
#         self._create_variables(n_features, n_classes)
#
#         self.model_output = tf.nn.softmax(tf.matmul(self.input_data, self.W_) + self.b_)
#
#         self._create_cost_function_node(self.loss_func, self.model_output, self.input_labels)
#         self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)
#         self._create_test_node()
#
#         self.obj_grad = tf.gradients(self.obj_function, self.input_data)[0]
#     def _create_placeholders(self, n_features, n_classes):
#
#         """ Create the TensorFlow placeholders for the model.
#         :param n_features: number of features
#         :param n_classes: number of classes
#         :return: self
#         """
#
#         self.input_data = tf.placeholder("float", [None, n_features], name='x-input')
#         self.input_labels = tf.placeholder("float", [None, n_classes], name='y-input')
#
#     def _create_variables(self, n_features, n_classes):
#
#         """ Create the TensorFlow variables for the model.
#         :param n_features: number of features
#         :param n_classes: number of classes
#         :return: self
#         """
#
#         self.W_ = tf.Variable(tf.zeros([n_features, n_classes]), name='weights')
#         self.b_ = tf.Variable(tf.zeros([n_classes]), name='biases')
#
#     def _create_test_node(self):
#
#         """
#         :return:
#         """
#
#         with tf.name_scope("test"):
#             correct_prediction = tf.equal(tf.argmax(self.model_output, 1), tf.argmax(self.input_labels, 1))
#             self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
#             _ = tf.summary.scalar('accuracy', self.accuracy)
#
#     def fit(self, train_set, train_labels, validation_set=None, validation_labels=None, restore_previous_model=False):
#
#         """ Fit the model to the data.
#         :param train_set: Training data. shape(n_samples, n_features).
#         :param train_labels: Labels for the data. shape(n_samples, n_classes).
#         :param validation_set: optional, default None. Validation data. shape(n_validation_samples, n_features).
#         :param validation_labels: optional, default None. Labels for the validation data. shape(n_validation_samples, n_classes).
#         :param restore_previous_model:
#                     if true, a previous trained model
#                     with the same name of this model is restored from disk to continue training.
#         :return: self
#         """
#
#         with tf.Session() as self.tf_session:
#             self._initialize_tf_utilities_and_ops(restore_previous_model)
#             self._train_model(train_set, train_labels, validation_set, validation_labels)
#             self.tf_saver.save(self.tf_session, self.models_dir + self.model_name)
#
#     def _train_model(self, train_set, train_labels, validation_set, validation_labels):
#
#         """ Train the model.
#         :param train_set: training set
#         :param train_labels: training labels
#         :param validation_set: validation set
#         :param validation_labels: validation labels
#         :return: self
#         """
#
#         for i in range(self.num_epochs):
#
#             shuff = list(zip(train_set, train_labels))
#             np.random.shuffle(shuff)
#
#             batches = [_ for _ in utilities.gen_batches(shuff, self.batch_size)]
#
#             batch = batches[0]
#             batch_xs, batch_ys = zip(*batch)
#             batch_xs = np.array(list(batch_xs))
#             batch_ys = np.array(list(batch_ys))
#             for batch_idx in range(len(batches) - 1):
#                 max_val, min_val = np.max(batch_xs), np.min(batch_xs)
#                 print(f"Max value in x_train {max_val}")
#                 pixels = {'max': max_val, 'min': min_val}
#
#                 cost_val, obj_function_val, obj_grads_val = self.tf_session.run(
#                     [self.cost, self.obj_function, self.obj_grad],
#                     feed_dict={self.input_data: batch_xs, self.input_labels: batch_ys})
#                 print(len(batches)*i + batch_idx,cost_val, obj_function_val)
#                 if np.isinf(cost_val).any() or np.isnan(cost_val).any():
#                     end_time = datetime.now()
#                     print("INFO: NaN found!")
#                     print(f"INFO: Time cost {end_time - self.start_time}\n")
#                     sys.exit(0)
#
#                 batch_xs, self.update_cnt, self.clip_cnt,self.useful_update_cnt = Utils.iterative_grist(batch_xs, obj_grads_val,
#                                      pixle_boundary=pixels,
#                                      update_stragety=update_stragety,
#                                      update_list=self.update_cnt,
#                                      clip_list=self.clip_cnt,
#                                      batch_size=self.batch_size,
#                                      useful_update_list=self.useful_update_cnt)
#
#                 self.iter_cnt = [x + 1 for x in self.iter_cnt]
#                 scores = [get_score(i, clips=self.clip_cnt, updates=self.update_cnt, iters=self.iter_cnt,useful_updates=self.useful_update_cnt) for i in range(self.batch_size)]
#                 scores_rank = list(np.argsort(np.array(scores)))[:self.bottom_k]
#
#                 batch_xs = np.clip(batch_xs, pixels['min'], pixels['max'])
#
#                 self.tf_session.run(self.train_step, feed_dict={self.input_data: batch_xs, self.input_labels: batch_ys})
#                 # change batch data
#                 self.safe_ptr += 1
#                 if self.safe_ptr % 100 == 0:
#                     new_batch = batches[batch_idx+1]
#                     new_batch_xs, new_batch_ys = zip(*new_batch)
#                     new_batch_xs = np.array(list(new_batch_xs))
#                     new_batch_ys = np.array(list(new_batch_ys))
#                     for new_i, image_idx in enumerate(scores_rank):
#                         batch_xs[image_idx] = new_batch_xs[new_i].copy()
#                         batch_ys[image_idx] = new_batch_ys[new_i].copy()
#                         self.iter_cnt[image_idx] = 0
#                         self.clip_cnt[image_idx] = 0
#                         self.update_cnt[image_idx] = 0
#                         self.useful_update_cnt[image_idx] = 0
#
#             if validation_set is not None:
#                 self._run_validation_error_and_summaries(i, validation_set, validation_labels)
#
#     def _run_validation_error_and_summaries(self, epoch, validation_set, validation_labels):
#
#         """ Run the summaries and error computation on the validation set.
#         :param epoch: current epoch
#         :param validation_set: validation set
#         :param validation_labels: validation labels
#         :return: self
#         """
#
#         feed = {self.input_data: validation_set, self.input_labels: validation_labels}
#         result = self.tf_session.run([self.tf_merged_summaries, self.accuracy], feed_dict=feed)
#         summary_str = result[0]
#         acc = result[1]
#
#         self.tf_summary_writer.add_summary(summary_str, epoch)
#
#         if self.verbose == 1:
#             print("Accuracy at step %s: %s" % (epoch, acc))
#
#     def predict(self, test_set, test_labels):
#
#         """ Compute the accuracy over the test set.
#         :param test_set: Testing data. shape(n_test_samples, n_features).
#         :param test_labels: Labels for the test data. shape(n_test_samples, n_classes).
#         :return: accuracy
#         """
#
#         with tf.Session() as self.tf_session:
#             self.tf_saver.restore(self.tf_session, self.models_dir + self.model_name)
#             return self.accuracy.eval({self.input_data: test_set, self.input_labels: test_labels})
