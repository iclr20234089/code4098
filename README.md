This is the implementation of Effective Offline Reinforcement Learning via Conservative State Value Estimation
We largely follow the implementaion in [d3rlpy](https://github.com/takuseno/d3rlpy/tree/61e6c00adc3fd0548e83500cc537ee0d27939a3f)

## installation
Check **Install form source## in [Installation of d3rlpy](https://d3rlpy.readthedocs.io/en/v1.1.1/installation.html)

## quick start
When using our code, you should first run
```
python dynamics.py
```
in d3rlpy/reproductions/offline directory to get the trained dynamics models.
After replacing the directories of dynamics models, you can run 
```
python csve_final_adroit.py 
```
in d3rlpy/reproductions/offline directory for training.

You should adjust the hyperparameters in the arguments to get the results in our paper.
## logging
The detailed results will be stored in the logs_09 directory. One can also view the results in directory exp_09 using tensorboard. 





