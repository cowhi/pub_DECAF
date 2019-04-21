import collections
import sys

import tensorflow as tf


def prepare_model_template(params):
    """Prepares the model template according to the settings in the
    parameter dictionary.

    Parameters
    ----------
    params : dict
        Directory that contains all general agent settings.
    
    Returns
    -------
    model_template : Function
        Returns a function that defines the model template as
        initialized with the parameter specifications.
    """
    if params['model'] == 'SimpleDQN':
        return SimpleDQN_template
    elif params['model'] == 'SimpleDQNFlat':
        return SimpleDQNFlat_template
    elif params['model'] == 'OriginalDQN':
        return DQN_template
    else:
        print('Model template "{}" is not defined!'.format(params['model']))
        sys.exit()


def DQN_template(state, name, output_shape, trainable, seed=42):
    """Builds the convolutional network used to compute the agent's Q-values
    and the weights of the importance network.

    Args:
        state: `tf.Tensor`, contains the agent's current state.
        name: Describes the output of the network.
        output_shape: Defines the number of output nodes of the network.
        trainable: makes the network parameters trainable or not.

    Returns:
        net: _network_type object containing the tensors output by the network.
    """
    net = tf.cast(state, tf.float32)
    net = tf.div(net, 255.)  # normalize between 0 and 1
    #net = Lambda(lambda x: (x - 127.5) / 127.5)(net)  # normalize between -1 and 1
    net = tf.layers.Conv2D(
        filters=32,
        kernel_size=8,
        strides=4,
        padding='same',
        kernel_initializer=tf.contrib.layers.xavier_initializer(seed=seed),
        activation=tf.nn.relu,
        trainable=trainable)(net)
    net = tf.layers.Conv2D(
        filters=64,
        kernel_size=4,
        strides=2,
        padding='same',
        kernel_initializer=tf.contrib.layers.xavier_initializer(seed=seed),
        activation=tf.nn.relu,
        trainable=trainable)(net)
    net = tf.layers.Conv2D(
        filters=64,
        kernel_size=3,
        strides=1,
        padding='same',
        kernel_initializer=tf.contrib.layers.xavier_initializer(seed=seed),
        activation=tf.nn.relu,
        trainable=trainable)(net)
    net = tf.layers.Flatten()(net)
    net = tf.layers.Dense(
        units=512,
        kernel_initializer=tf.contrib.layers.xavier_initializer(seed=seed),
        activation=tf.nn.relu,
        trainable=trainable)(net)
    output = tf.layers.Dense(
        units=output_shape,
        kernel_initializer=tf.contrib.layers.xavier_initializer(seed=seed),
        activation=None,
        trainable=trainable)(net)
    return collections.namedtuple('DQN', [name])(output)

def SimpleDQN_template(state, name, output_shape, trainable, seed=42):
    """Builds the convolutional network used to compute the agent's Q-values.

    Args:
      state: `tf.Tensor`, contains the agent's current state.

    Returns:
      net: _network_type object containing the tensors output by the network.
    """
    flat = tf.layers.Flatten()(state)
    fc1 = tf.layers.Dense(
        64,
        activation=tf.nn.relu,
        kernel_initializer=tf.contrib.layers.xavier_initializer(seed=seed),
        trainable=trainable)(flat)
    fc2 = tf.layers.Dense(
        64,
        activation=tf.nn.relu,
        kernel_initializer=tf.contrib.layers.xavier_initializer(seed=seed),
        trainable=trainable)(fc1)
    fc3 = tf.layers.Dense(
        64,
        activation=tf.nn.relu,
        kernel_initializer=tf.contrib.layers.xavier_initializer(seed=seed),
        trainable=trainable)(fc2)
    output = tf.layers.Dense(
        output_shape, 
        activation='linear',
        kernel_initializer=tf.contrib.layers.xavier_initializer(seed=seed),
        trainable=trainable)(fc3)
    return collections.namedtuple('SimpleDQN', [name])(output)

def SimpleDQNFlat_template(state, name, output_shape, trainable, seed=42):
    """Builds the convolutional network used to compute the agent's Q-values.

    Args:
      state: `tf.Tensor`, contains the agent's current state.

    Returns:
      net: _network_type object containing the tensors output by the network.
    """
    flat = tf.layers.Flatten()(state)
    fc1 = tf.layers.Dense(
        64,
        activation=tf.nn.relu,
        kernel_initializer=tf.contrib.layers.xavier_initializer(seed=seed),
        trainable=trainable)(flat)
    output = tf.layers.Dense(
        output_shape, 
        activation='linear',
        kernel_initializer=tf.contrib.layers.xavier_initializer(seed=seed),
        trainable=trainable)(fc1)
    return collections.namedtuple('SimpleDQNFlat', [name])(output)