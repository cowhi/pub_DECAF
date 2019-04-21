# pub_DECAF

Source code and results for the DECAF research paper

## Installation

### System requirements

This software has been tested on MacOS Mojave, Ubuntu 16.04 LTS, and CentOS 7. The following packages are needed to run the software (I recommend installation with conda):

- Python 3.6
- Tensorflow 1.10.0 - 1.12.0
- OpenCV >3.4
- pyyaml >5.1
- gym
- gym[atari]

If you use conda, get the proper tensorflow and openCV versions like this:
    
    `conda create -n decaf python=3.6`
    
    `source activate decaf` or `conda activate decaf`
    
    `conda install -c anaconda tensorflow-gpu=1.12`
    
    `conda install -c anaconda opencv`
    
    `pip install gym`
    
    `pip install gym[atari]`

### Download

- Download the software:

    `git clone git@github.com:cowhi/pub_DECAF.git`

### Test installation

**Important**: This software creates directories and generates logfiles in the repo base directory. Please run the experiments only from within the `pub_DECAF/src` directory.

Learn the CartPole-v0 environment from the classic domain in OpenAI gym:

    `python main.py --params paramsCartPole-v0.yaml`

Perform tests in an environment (set parameter 'init_weights' to use pretrained agent and test performance, otherwise random weights):

    `python main.py --params paramsAtariDQN.yaml --mode test --episodes 100`

Just watch an agent perform in an environment (set parameter 'init_weights' to use pretrained agent):

    `python main.py --params paramsAtariDQN.yaml --mode render`

## Running experiments

In general, I recommend running experiments like this:

    `python main.py --params paramsAtariDQN.yaml >experiment.log 2>&1 &`

And then monitor the output like this

    `tail -f experiment.log`

### Experiment parameter

All parameters can be set in the parameter file found in `pub_DECAF/params`. The existing files are best examples for certain experiments and environments but they can be adapted as necessary.

For example, it often makes sense to run multiple experiments at the same time. This can be set with the `runs` paramter. Another example would be which algorithm to use (`agent_type`) or which network architecture (`model`).

### Logfiles

Every experiment generates logfiles to save the results but also checkpoints during training so it is possible to pick up training after a crash or reuse the network parameters in later experiments or for just showing what has been learned. Those logs are found in the `pub_DECAF/logs` directory sorted by kind of experiment and then date.

## Visualization of experiments and comparison

TODO

### For debugging the algorithm and detailed evaluation

- Start tensorboard:

    `tensorboard --logdir "pub_DECAF/logs/[Domain]/[Environment]" --host 0.0.0.0 --port 6006 &`

- Goto to `http://0.0.0.0:6006` in browser and check the various graphs or the network architecture.

### Visualization of results

- Use `pub_DECAF/notebooks/visulaization.ipynb` to visualize results for individual runs and summaries of runs for each algorithm.

## Influences

The following is a list of works that had a very strong influence on the code and were often used to copy some sections entirely:

- https://github.com/keras-rl
- https://github.com/google/dopamine
