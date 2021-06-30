import os
import sys
from datetime import datetime
"""Parser of command args"""
import argparse

parse = argparse.ArgumentParser()
parse.add_argument("--type", type=str,choices=['origin', 'grist',], help="run initial file or grist file")
flags, unparsed = parse.parse_known_args(sys.argv[1:])

case_dict = {
    "GH_IPS1":["scripts.study_case.GH_IPS1.ghips1"],
    "GH_IPS1_mutant":["scripts.study_case.GH_IPS1_mutant.ghips1_mutant"],
    "GH_IPS6":["scripts.study_case.GH_IPS6.command_line.run_logistic_regression"],
    "GH_IPS9":["scripts.study_case.GH_IPS9.ghips9"],
    "handon_tensorflow":["scripts.study_case.handon_tensorflow.code_10_image"],
    "MNIST":["scripts.study_case.MNIST.v3.train"],
    "python_tensorflow":["scripts.study_case.python_tensorflow.mnist"],
    "SO_IPS1":["scripts.study_case.SO_IPS1.soips1"],
    "SO_IPS2":["scripts.study_case.SO_IPS2.soips2"],
    "SO_IPS6":["scripts.study_case.SO_IPS6.soips6"],
    "SO_IPS7":["scripts.study_case.SO_IPS7.soips7"],
    "SO_IPS14":["scripts.study_case.SO_IPS14.soips14"],
    "tensorflow_examples":["scripts.study_case.tensorflow_examples.logistic_regression"],
    "tensorflow_examples_tutorials_mnist":["scripts.study_case.tensorflow_examples_tutorials_mnist.mnist_softmax"],
    "Tensorflow_gesture_Demo":["scripts.study_case.Tensorflow_gesture_Demo.Mnist"],
    "tensorflow_in_ml":["scripts.study_case.tensorflow_in_ml.softmax"],
    "tensorflow_mnist":["scripts.study_case.tensorflow_mnist.mnist"],
    "TensorFuzz":["scripts.study_case.TensorFuzz.nan_model_exp",
                  "scripts.study_case.TensorFuzz.nan_model_truediv",
                  "scripts.study_case.TensorFuzz.nan_model_log"],
    "tensorflow_value_iteration_networks_v1":["scripts.study_case.tensorflow_value_iteration_networks_v1.train"],
    "generative_models_v1":["scripts.study_case.generative_models_v1.GAN.infogan.infogan_tensorflow"],
    "generative_models_v2":["scripts.study_case.generative_models_v2.GAN.auxiliary_classifier_gan.ac_gan_tensorflow",
                            "scripts.study_case.generative_models_v2.GAN.ali_bigan.ali_bigan_tensorflow",
                            "scripts.study_case.generative_models_v2.GAN.boundary_seeking_gan.bgan_tensorflow",
                            "scripts.study_case.generative_models_v2.GAN.coupled_gan.cogan_tensorflow",
                            "scripts.study_case.generative_models_v2.GAN.disco_gan.discogan_tensorflow",
                            "scripts.study_case.generative_models_v2.GAN.mode_regularized_gan.mode_reg_gan_tensorflow",
                            "scripts.study_case.generative_models_v2.GAN.vanilla_gan.gan_tensorflow"],
    "tf_unet":["scripts.study_case.tf_unet.test_toy"],
    "pytorch_playground":["scripts.study_case.pytorch_playground.pytorch_pg"],
    "SC_DNN":["scripts.study_case.SC_DNN.sc_train_creg",
              "scripts.study_case.SC_DNN.sc_train_creg_div2",
              "scripts.study_case.SC_DNN.sc_train_l2reg",
              "scripts.study_case.SC_DNN.sc_train_l2reg_div2"],
    "skorch":["scripts.study_case.skorch.main"],
    "RBM_grist":["scripts.study_case.RBM_grist.rbm"],
    "pytorch_geometric_exp":["scripts.study_case.pytorch_geometric_exp.test.utils.test_softmax"],
    "pytorch_geometric_fork":["scripts.study_case.pytorch_geometric_fork.test.nn.models.test_autoencoder"],
    "Matchzoo":["scripts.study_case.MatchZoo_py.tests.test_losses"],
    "MachineLearning":["scripts.study_case.MachineLearning.temp"],
    "DeepLearningTest":["scripts.study_case.DeepLearning.deeplearningtest"],
    "tensorflow_GAN_MNIST":["scripts.study_case.tensorflow_GAN_MNIST.GAN_MNIST"],
    "gan_practice":["scripts.study_case.gan_practice.gan_mnist"],
    "CS231":["scripts.study_case.CS231.assign3_acgan",
             "scripts.study_case.CS231.assign3_acgan_log1",
             "scripts.study_case.CS231.assign3_acgan_log2"],
    "deep_learning_Nikolenko_and_Co":["scripts.study_case.deep_learning_Nikolenko_and_Co.ch10_04_01",
                                      "scripts.study_case.deep_learning_Nikolenko_and_Co.ch10_04_03_Pic_10_05",
                                      "scripts.study_case.deep_learning_Nikolenko_and_Co.ch10_04_04_Pic_10_06",
                                      "scripts.study_case.deep_learning_Nikolenko_and_Co.ch10_04_05_Pic_10_07",
                                      "scripts.study_case.deep_learning_Nikolenko_and_Co.ch10_04_06_Pic_10_08"],
    "FuzzForTensorflow":["scripts.study_case.nan_model"],
    "git1_rbm":["scripts.study_case.My_pytorch1"],
    "gongdols":["scripts.study_case.denoising_RBM",
                "scripts.study_case.main"],
    "MNIST_DCGAN":["scripts.study_case.code",
                   "scripts.study_case.code_log1",
                   "scripts.study_case.code_log2"],
    "softmax_gan":["scripts.study_case.softmax_gan.softmax_gan_tensorflow"],
}
cur_cnt = 1
run_cnt = 1
s1 = datetime.now()
special_dict = {"tf_unet":"tensorflow0120",
                "pytorch_playground":"pytorch151",
                "skorch": "pytorch151",
                "RBM_grist": "pytorch151",
                "pytorch_geometric_exp":"pytorch151",
                "pytorch_geometric_fork":"pytorch151",
                "Matchzoo":"pytorch151",}

for case_repo,commands in case_dict.items():
    print(f"INFO: Executing {case_repo}. {cur_cnt} of {len(case_dict)}")
    for cm in commands:
        cm = cm+"_grist" if flags.type == "grist" else cm
        if case_repo in special_dict.keys():
            env_name = special_dict[case_repo]
        else:
            env_name = "tensorflow181"

        python_command = f"/root/anaconda3/envs/{env_name}/bin/python -u -m {cm}"

        status1 = os.system(python_command)
        print(f"INFO: {cm}")
        if status1 == 0:
            print(f"INFO: Execution Run {cm} finished!")
        else:
            print(f"ERROR: Fail to run {cm}")

        print(f"INFO: {case_repo} is Over!")
        run_cnt += 1
    cur_cnt += 1
print(f"Done! Total Time cost: {datetime.now() - s1}")