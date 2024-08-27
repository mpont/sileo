<h1>SILEO</span></h1>

Official implementation of [Self-Improving Logic from Experimental Observations] by Max Pont.
Derived from the original [[TD-MPC2]](https://arxiv.org/abs/2310.16828) [[implementation]](https://github.com/nicklashansen/tdmpc2).

----

## Overview

SILEO is a framework that explores a novel representation of low-level actions using high-dimensional rotations of the observation space. This representation of actions using their dynamics opens up a number of promising applications, from machine understanding to unsupervised hierarchical learning and inference time reduction.
This repository allows for online training of a TDMPC2-derived agent on a variety of tasks. As in our paper, only DMControl tasks have been tested and are currently supported, however it should be possible to use this code for all online datasets used by the original agent with minimal overhead. 

## Installation

This code was built using Ubuntu for WSL2.0 as an operating system, and will likely not work under any other subsystem due to certain libraries required by the gym environments. The environment is the same as the original agent, please refer to the [[original repository]](https://github.com/nicklashansen/tdmpc2) for an installation guide. We recommend however using the conda environment for setup and training instead of a Docker container.

## Modified and added features

The new classes used in our paper are found under the sileo folder. Action and its subclass CompositeAction implement the action representation described in our paper as well as the methods required to update them. Suspension implements various projections used on the observation or latent state spaces, both when exploring their strangths and during our experiments. 

The original TDMPC2 world model (found in the common folder) has been left largely unchanged; the next method has been modified however to accomodate Actions of arrays of Actions as well as the original embeddings tensors. 

The trainer and online trainer class have been modified to allow for the rotations repreenting the actions used to be updated alongside the rest of the model. They support the training of two consecutive online tasks, using for instance the following code:
```
$ python train.py task=dog-run +second_task=dog-walk steps=600000
```
Most of the changes were made directly to the TDMPC2 (or rather DRQV2) agent to improve readability. Several new methods were added, notably think and prune which implement the pruning algorithm alongside their auxiliary functions such as cluster. The original methods were also modified to implement planning, action and reward prediction using a finite action set after pre-training as well as to allow for the actions used to be updated when relevant. The model updates are made by casting all past actions, even in the auxiliary continuous environments, as Actions with their pre-defined embeddings. 
