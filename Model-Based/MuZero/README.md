# MuZero & Connect4
This repo is a (simplified) Python implementation of DeepMind's MuZero algorithm based on the work of Johan Gras https://github.com/johan-gras/MuZero, which in turn was based on
the extensive Python pseudocode provided by DeepMind's MuZero team.
Modifications have been made to apply MuZero to play Connect4, as was done for AlphaZero by Soh Wee Tee https://github.com/plkmo/AlphaZero_Connect4.
This was done as part of an investigation into state-of-the-art model-based reinforcement learning algorithms as part of my honours degree.

See the following papers and articles for detailed theory and explanation for choices of algorithm design.
__ https://arxiv.org/abs/1911.08265
__ https://arxiv.org/src/1911.08265v1/anc/pseudocode.py
__ https://gym.openai.com/envs/CartPole-v1/

To apply these algorithms please ensure that the following dependencies are correctly configured in your Python environment.
- Conda **4.7.12**
- Python **3.7**
- Tensorflow **2.0.0**
- Numpy **1.17.3**

To run MuZero on your environment simply follow the following steps:
1. Create configuration of MuZero in ``config.py``.
2. Make a function call to the configuration inside the main function of ``muzero.py``.
3. Run the main function: ``python muzero.py`` in your terminal.

### Creating your own environment

1. Create a class that extends ``AbstractGame``, this class should implement the behaviour of your environment.
For instance, the ``CartPole`` class extends ``AbstractGame`` and works as a wrapper upon `gym CartPole-v1`__.
You can use the ``CartPole`` class as a template for any gym environment.

__ https://gym.openai.com/envs/CartPole-v1/

2. **This step is optional** (only if you want to use a different kind of network architecture or value/reward transform).
Create a class that extends ``BaseNetwork``, this class should implement the different networks (representation, value, policy, reward and dynamic) and value/reward transforms.
For instance, the ``CartPoleNetwork`` class extends ``BaseNetwork`` and implements fully connected networks.

3. **This step is optional** (only if you use a different value/reward transform).
You should implement the corresponding inverse value/reward transform by modifying the ``loss_value`` and ``loss_reward`` function inside ``training.py``.

### Differences from the paper

As noted by Gras, this implementation differ from the original paper in the following manners:

- We use fully connected layers instead of convolutional ones. This is due to the nature of our environment (Cartpole-v1) which as no spatial correlation in the observation vector.
- We don't scale the hidden state between 0 and 1 using min-max normalization. Instead we use a tanh function that maps any values in a range between -1 and 1.
- We do use a slightly simple invertible transform for the value prediction by removing the linear term.
- During training, samples are drawn from a uniform distribution instead of using prioritized replay.
- We also scale the loss of each head by 1/K (with K the number of unrolled steps). But, instead we consider that K is always constant (even if it is not always true).
