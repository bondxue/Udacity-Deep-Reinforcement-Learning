[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Project 1: Navigation

[![Project passed](https://img.shields.io/badge/project-passed-success.svg)](https://img.shields.io/badge/project-passed-success.svg)

### Introduction

For this project, I have trained an agent to navigate (and collect bananas!) in a large, square world. The orginal setup of this project is to be found [here](https://github.com/udacity/deep-reinforcement-learning/tree/master/p1_navigation).

![Trained Agent][image1]

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

### Setup

1. The necessary banana environment to run the project can be downloaded from one of the links below for different operating systems:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

2. Place the file in the [DRLND GitHub repository](https://github.com/udacity/deep-reinforcement-learning), in the `p1_navigation/` folder, and unzip (or decompress) the file. 

3. The Python environment setup can be found [here](https://github.com/udacity/deep-reinforcement-learning#dependencies).

### Project Structure

+ `Navigation.ipynb`- contains the code to set up the environment and the outer episode iteration to solve the reinforcement problem. Our solution uses a (double) deep Q-learning network (only standard feedforward layers) and experience replay.

+ `dqn_agent.py` - contains the **Agent** and **ReplayBuffer** classes. The **Agent** class can choose mode between a *standard DQN* or a *Double DQN*, and **ReplayBuffer** is used as the replay memory to store and recall experience tuples.

+ `model.py` - contains the **QNetwork** class which is to structure our deep Q-network architecture. 

+ `Report.pdf` - contains the detailed description of implementation and result discussion. 


