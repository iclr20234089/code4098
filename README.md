This is the original code for the ICLR2023 submission *Effective Offline Reinforcement Learning via Conservative State Value Estimation*.
We largely follow the implementaion in [d3rlpy](https://github.com/takuseno/d3rlpy/commit/1ac85b9955408b5e3e9f67ec6592828d8021885b).

## Installation
Check **Install form source** in [Installation of d3rlpy](https://d3rlpy.readthedocs.io/en/v1.1.1/installation.html)

## A Brief Mannual
To reproduce any experiments with our proposed algorithm, you should first train the dynamic model and then train the offline rl models. 

1. **Learn and store the dynamic model.** You should run
```
python dynamics.py
```
in d3rlpy/reproductions/offline directory to get the trained dynamics models.

2. **Train and evaluate with our algorithm.** After replacing the directories of dynamics models, you can start trainning through 
```
python csve_final_adroit.py 
```
in d3rlpy/reproductions/offline directory.

You should adjust the hyperparameters in the arguments to get the results in our paper.

## logging
The detailed results will be stored in the logs_09 directory. One can also view the results in directory exp_09 using tensorboard. 

For any questions, you can post them in the openreview thread under our submission.



