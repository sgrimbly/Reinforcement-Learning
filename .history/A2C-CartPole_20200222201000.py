# A2C CartPole Implemented using tutorial below
# http://inoryy.com/post/tensorflow2-deep-reinforcement-learning/#advantage-actor-critic-with-tensorflow-2-1

import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as kl

class ProbabilityDistribution(tf.keras.Model):
  def call(self, logits, **kwargs):
    # Sample a random categorical action from the given logits.
    return tf.squeeze(tf.random.categorical(logits, 1), axis=-1)
    
