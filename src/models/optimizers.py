import sys

import tensorflow as tf

def prepare_optimizer(params):
    """Prepares the model optimizer according to the settings in the
    parameter dictionary.

    Parameters
    ----------
    params : dict
        Directory that contains all general agent settings.
    
    Returns
    -------
    optimizer : tf.train.Optimizer
        Optimizer as initialized with the parameter specifications.
    """
    if params['optimizer'] == 'Adam':
        return tf.train.AdamOptimizer(
            learning_rate=params['alpha'],
            beta1=params['adam_beta1'],
            beta2=params['adam_beta2'],
            epsilon=params['optimizer_epsilon'],
            use_locking=params['optimizer_locking'],
            name=params['optimizer'])
    elif params['optimizer'] == 'RMSProp':
        return tf.train.RMSPropOptimizer(
            learning_rate=params['alpha'],
            decay=params['rms_decay'],
            momentum=params['momentum'],
            epsilon=params['optimizer_epsilon'],
            use_locking=params['optimizer_locking'],
            centered=params['rms_centered'],
            name=params['optimizer'])
    else: 
        print('Optimizer "{}" is not defined!'.format(params['optimizer']))
        sys.exit()
