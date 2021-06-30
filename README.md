# Exposing Numerical Bugs in Deep Learning via Gradient Back-Propagation

This is the implementation repository of our incoming ESEC/FSE 2021 paper:  **Exposing Numerical Bugs in Deep Learning via Gradient Back-Propagation** 

## Description

In this paper, we propose a novel bug finding technique that automatically generates a small input that exposes numerical bugs by inducing NaN or INF values called GRIST (**GR**ad**I**ent **S**earch based Numerical Bug **T**riggering). The technique piggy-backs on the built-in gradient computation functionalities of deep learning platforms. Our evaluation on 63 real-world subjects  containing 79 bugs shows that our technique  detects 78 numerical bugs including 33 that cannot be exposed by the inputs provided in the projects and 31 that cannot be detected by the state-of-the-art technique named DEBAR. Meanwhile, our technique  can save the time cost of triggering bugs by 8.79 times on average. Our tool and datasets are released here. 

## Datasets

In total, we collected 63 DL programs with 79 numerical bugs (each DL program contains at least one numerical bugs) as subjects from the following two sources: 

- **Known bugs from existing studies and GitHub**: We used 17 subjects containing 23 known bugs from existing studies and GitHub. 
  Specifically, we used 8 subjects from the [existing empirical study](https://github.com/ForeverZyh/TensorFlow-Program-Bugs) on TensorFlow program bugs and one subject from [TensorFuzz](https://github.com/brain-research/tensorfuzz).
  Regarding known bugs from GitHub, we adopted bug-relevant keywords (including NaN, INF, and the operations listed in paper) to search a set of candidate programs from GitHub according to the descending order of GitHub searching relevance and then conducted manual filtering. we used 8 subjects whose 10 bugs can be reproduced conveniently and successfully in our runtime experiment.
- **Unknown bugs from GitHub**: We applied GRIST and the state-of-the-art technique DEBAR to fuzz GitHub DL programs and finally identified 46 subjects with 56 unknown numerical bugs to developers.

All 63 programs of our paper can be found in `scripts/study_case` and the correspondence between their ID and project name is in [dataset_info.md](dataset_info.md).

## Experiment

📣 ***NOTE: In order to provide an available and functional artifact for our paper, two authors are responsible for refactoring the code and providing an out-of-the-box docker image before the official publication. (Work In Progress)***

### Installation

**Step 0. Install Anaconda**

Please follow this [document](https://docs.anaconda.com/anaconda/install/) to install Anaconda (a python package management software) 

**Step 1. Create Virtual Environments**

 ```shell
conda create -n your_python_env python=3.6 
conda activate your_python_env
 ```

**Step 3. Run each case like this:**

```shell
cd scripts/study_case/ID_X
python -u ID_X_SCRIPTS_grist.py
```

### Configuration

There are some hyper-parameters in GRIST and they could be easily configured as follows.

` In experiments.conf`

```
[parameter]
# time limitation (minutes)
time_limit=30
# epsilon in paper
change=0.15
# The choice of running grist or default input
delta_type=grist
# The choice of running grist or grist_ns
switch_data=True
# rate of updating data in each iteration
drop_rate=0.05
```

## Hardware and Runtime Environments:

We used the Anaconda environments to switch between different versions of PyTorch and TensorFlow. Our study was conducted on the Intel Xeon Silver 4214 machine with 128GB RAM, Ubuntu 16.04.6 LTS, and two GTX 2080 Ti GPUs. 
