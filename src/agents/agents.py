import collections
import os
import random
import sys

import numpy as np
import tensorflow as tf
from tensorflow.python.pywrap_tensorflow import NewCheckpointReader 

sys.path.append('..')
from memories import prepare_memory  # pylint: disable=import-error
from models import (  # pylint: disable=import-error
    prepare_model_template, 
    prepare_optimizer
)
#from policies import prepare_policy


def prepare_agent(params, private_params, private_paths, sess, summary_writer):
    """Prepares an agent according to the settings in the parameter
    dictionary.

    Parameters
    ----------
    params : dict
        Directory that contains all general agent settings.
    privat_params : dict
        Directory that contains all specific agent settings.
    sess : tf.Session
        Session that is used throughout the experiment
    summary_writer : tf.summary.FileWriter
        Needed to log tensorflow information during training.

    Returns
    -------
    agent : Agent
        Agent as initialized with the parameter specifications.
    """
    if params['agent_type'] == 'DQN':
        return DQNAgent(
            params=params, 
            private_params=private_params,
            private_paths=private_paths,
            sess=sess,
            summary_writer=summary_writer)
    elif params['agent_type'] == 'A2T':
        return A2TAgent(
            params=params, 
            private_params=private_params,
            private_paths=private_paths,
            sess=sess,
            summary_writer=summary_writer)
    elif params['agent_type'] == 'BaseLine':
        return BaselineAgent(
            params=params, 
            private_params=private_params,
            private_paths=private_paths,
            sess=sess,
            summary_writer=summary_writer)
    else:
        print('Agent "{}" not implemented'.format(params['agent_type']))
        sys.exit()

def linearly_decaying_factor(decay_period, step, warmup_steps, value):
    """Returns the current value for a decaying factor.

    This follows the Nature DQN schedule of a linearly decaying epsilon (Mnih et
    al., 2015). The schedule is as follows:
        Begin at 1. until warmup_steps steps have been taken; then
        Linearly decay epsilon from 1. to epsilon in decay_period steps; and then
        Use epsilon from there on.

    Parameters
    ----------
        decay_period : float
            Period over which epsilon is decayed.
        step : int
            Number of training steps completed so far.
        warmup_steps : int
            Number of steps taken before epsilon is decayed.
        value : float
            Final value to which to decay the factor.

    Returns
    -------
        value : float
            Current value computed according to the schedule.
    """
    steps_left = decay_period + warmup_steps - step
    bonus = (1.0 - value) * steps_left / decay_period
    bonus = np.clip(bonus, 0., 1. - value)
    return value + bonus


class Agent(object):
    """ Base class for agent implementations """
    def __init__(
            self,
            params, 
            private_params, 
            private_paths,
            sess,
            summary_writer=None):
        self.name = 'Agent'
        self.params = params
        self.private_params = private_params
        self.private_params['train_step'] = 0
        self.private_params['agent_step'] = 0
        self.private_params['target_updates'] = 0
        self.private_paths = private_paths

        self.get_epsilon = linearly_decaying_factor

        self.sess = sess
        self.summary_writer=summary_writer
        # Initialize the agent
        with tf.device(self.private_params['device']):
            # build the replay memory
            self.memory = prepare_memory(self.params, self.private_params)
            # build the optimzer
            self.optimizer = prepare_optimizer(self.params)
            # Build the model template
            self.model_template = prepare_model_template(self.params)
            #TODO: Build agent policy
            #self.policy = prepare_policy(self.params, self.private_params)
            # Define placeholder for the network input 
            self.state_ph = tf.placeholder(
                shape=self.private_params['state_shape'],
                dtype=self.private_params['observation_dtype'],
                name='state_ph')
            # Make a dummy state representation
            self.state = np.zeros(self.private_params['state_shape'])
            # Actually build the networks
            self.build_networks()
            # Define the training operations for the network
            self.train_op = self.build_train_op()  # pylint: disable=E1111
            # Define the sync operations between networks
            self.sync_ops = self.build_sync_ops()  # pylint: disable=E1111
        # Combine all summaries
        if self.summary_writer is not None:
            self.merged_summaries = tf.summary.merge_all()
        # TODO: Load pretrained agents in network
        #self.load_pretrained_weights()
        #if self.params['init_weights'] is not None:
        #    self.init_weights()
        # Inititialize the checkpoint saver and limit checkpoints to
        # to save memory
        if self.params['mode'] is not 'render':
            self.checkpoint_saver = tf.train.Saver(
                max_to_keep=self.params['checkpoints_keep_max'])
        # Prepare variables that the agent needs to keep track on
        self.observation = None
        self.last_observation = None


    def build_networks(self):
        pass

    def build_train_op(self):
        pass
    
    def build_train_target(self):
        pass

    def build_sync_ops(self):
        pass

    def load_pretrained_weights(self):
        # TODO: When implementing A2T
        pass
  
    def variable_summaries(self, var, name):
        """Attach a lot of summaries to a Tensor for evaluation and
        visualization in TensorBoard.
        
        Parameters
        ----------
        var : np.array
            Array of variables that need to be analized.
        name : str
            Name of the varible that is analized here.
        """
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('{}Mean'.format(name), mean)
            with tf.name_scope('Std'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('{}Std'.format(name), stddev)
            tf.summary.scalar('{}Max'.format(name), tf.reduce_max(var))
            tf.summary.scalar('{}Min'.format(name), tf.reduce_min(var))
            tf.summary.histogram('{}Histogram'.format(name), var)

    def start_episode(self, observation, episode_type):
        """Initializes an episode.

        Parameters
        ----------
            observation: np.array
                Observation provided from the environment.
            episode_type : str
                Type of episode, either `train` or `eval`.

        Returns
        -------
            action : int
                Action ID of the selected action.
        """
        self.reset_state()
        self.record_observation(observation)
        if episode_type == 'train':
            self.perform_train_step()
            self.private_params['agent_step'] += 1
        self.action = self.select_action(episode_type)  # pylint: disable=E1111
        return self.action

    def step(self, reward, observation, episode_type):
        """Records the most recent transition and returns the agent's next action.

        We store the observation of the last time step since we want to store it
        with the reward.

        Parameters
        ----------
            reward: float
                Last reward from the environment.
            observation: np.array
                Observation provided from the environment.
            episode_type : str
                Type of episode, either `train` or `eval`.

        Returns
        -------
            action : int
                Action ID of the selected action.
        """
        self.last_observation = self.observation
        self.record_observation(observation)
        if episode_type == 'train':
            self.store_transition(self.last_observation, self.action, reward, False)
            self.perform_train_step()
            self.private_params['agent_step'] += 1
        self.action = self.select_action(episode_type)  # pylint: disable=E1111
        return self.action

    def end_episode(self, reward, episode_type):
        """Signals the end of the episode to the agent.

        We store the observation of the current time step, which is the last
        observation of the episode.

        Parameters
        ----------
            reward: float
                Last reward from the environment.
            episode_type : str
                Type of episode, either `train` or `eval`.
        """
        if episode_type == 'train':
            self.store_transition(self.observation, self.action, reward, True)


    def reset_state(self):
        self.state.fill(0)

    def record_observation(self, observation):
        """Records an observation and update state.

        Extracts a frame from the observation vector and overwrites the oldest
        frame in the state buffer.

        Parameters
        ----------
            observation: np.array
                Observation provided from the environment.
        """
        # Set current observation. We do the reshaping to handle environments
        # without frame stacking.
        observation = np.reshape(
            observation,
            self.private_params['observation_shape'])
        self.observation = observation[..., 0]
        self.observation = np.reshape(
            observation,
            self.private_params['observation_shape'])
        # Swap out the oldest frame with the current frame.
        self.state = np.roll(self.state, -1, axis=-1)
        self.state[0, ..., -1] = self.observation

    def perform_train_step(self):
        """Runs a single training step.

        Runs a training op if both:
        (1) A minimum number of frames have been added to the replay buffer.
        (2) `training_step` is a multiple of `update_period`.

        Also, syncs weights from online to target network if training steps is a
        multiple of target update period.
        """
        # Run a train op at the rate of self.update_period if enough training steps
        # have been run. This matches the Nature DQN behaviour.
        if self.memory.memory.add_count > self.params['min_memory_size']:
            # TODO: Check if we can solve this better. Now the steps are taken inside the
            # atari environment preprocessor and not actually by the agent
            if self.private_params['agent_step'] % self.params['train_frequency'] == 0:
                self.sess.run(self.train_op)
                self.private_params['train_step'] += 1
                if (self.private_params['train_step'] % self.params['summary_frequency'] == 0):
                    summary = self.sess.run(self.merged_summaries, {self.state_ph: self.state})
                    self.summary_writer.add_summary(summary, self.private_params['train_step'])
                    self.summary_writer.flush()
                    
                if self.private_params['train_step'] % self.params['target_update_frequency'] == 0:
                    #tf.logging.info('\t{} - Updating target networks: Step {}'.format(self.name, self.private_params['train_step']))
                    self.sess.run(self.sync_ops)
                    self.private_params['target_updates'] += 1

    def select_action(self, *args, **kwargs):
        pass

    def store_transition(self, last_observation, action, reward, is_terminal):
        """Stores an experienced transition.

        Executes a tf session and executes replay buffer ops in order to store the
        following tuple in the replay buffer:
            (last_observation, action, reward, is_terminal).

        Pedantically speaking, this does not actually store an entire transition
        since the next state is recorded on the following time step.

        Parameters
        ----------
        last_observation: np.array
            Last observation from the environment.
        action: int
            Action taken in the last observation.
        reward: float
            Reward received.
        is_terminal: bool
            Indicating if the current state is a terminal state.
        """
        self.memory.add(last_observation, action, reward, is_terminal)

    def unbundle(self, era_number, bundle_dictionary):
        """Restores the agent from a checkpoint.

        Restores the agent's Python objects to those specified in bundle_dictionary,
        and restores the TensorFlow objects to those specified in the
        checkpoint_dir. If the checkpoint_dir does not exist, will not reset the
        agent's state.

        Parameters
        ----------
            era_number : int
                Latest checkpoint version, used when restoring replay
                memory.
            bundle_dictionary: dict
                Dictionary that contains additional Python objects
                owned by the agent.

        Returns
        -------
            success : bool
                True if unbundling was successful.
        """
        try:
            # self.memory.load() will throw a NotFoundError if it does not find all
            # the necessary files, in which case we abort the process & return False.
            self.memory.load(
                self.private_paths['ckpt_dir'],
                era_number)
        except tf.errors.NotFoundError:
            return False
        for key in self.__dict__:
            if key in bundle_dictionary:
                #print(key)
                if key is not 'private_params':
                    self.__dict__[key] = bundle_dictionary[key]
        # Restore the agent's TensorFlow graph.
        self.checkpoint_saver.restore(
            self.sess,
            os.path.join(
                self.private_paths['ckpt_dir'],
                'tf_ckpt-{}'.format(era_number)))
        return True

    def bundle_and_checkpoint(self, era_number, checkpoint_dir):
        """Returns a self-contained bundle of the agent's state.

        This is used for checkpointing. It will return a dictionary containing all
        non-TensorFlow objects (to be saved into a file by the caller), and it saves
        all TensorFlow objects into a checkpoint file.

        Parameters
        ----------
            era_number : int
                Era number to use for naming the checkpoint file.
            checkpoint_dir : str
                Directory where TensorFlow objects will be saved.
            
        Returns
        -------
            bundle_dictionary : dict
                A dict containing additional Python objects to be checkpointed by the
                experiment. If the checkpoint directory does not exist, returns None.
        """
        #tf.logging.info('\t{} - Creating new checkpoint: Era {}, Step {}'.format(
        #    self.name,
        #    era_number,
        #    self.private_params['agent_step']))
        if not tf.gfile.Exists(checkpoint_dir):
            return None
        # Call the Tensorflow saver to checkpoint the graph.
        self.checkpoint_saver.save(
            self.sess,
            os.path.join(checkpoint_dir, 'tf_ckpt'),
            global_step=era_number)
        # Checkpoint the out-of-graph replay buffer.
        self.memory.save(checkpoint_dir, era_number)
        # Create additional agent data
        bundle_dictionary = {
            'state': self.state,
            'private_params': self.private_params
        }
        return bundle_dictionary

    def init_weights(self):
        """ Loads the weights of a pretrained agent into the network. """
        # Define variable mapping according to agent source
        if self.params['init_from'] in ['RL2go']:
            vars_mapping = {  
                'Online/conv2d/bias:0': 'Online/conv2d/bias', 
                'Online/conv2d/kernel:0': 'Online/conv2d/kernel', 
                'Online/conv2d_1/bias:0': 'Online/conv2d_1/bias', 
                'Online/conv2d_1/kernel:0': 'Online/conv2d_1/kernel', 
                'Online/conv2d_2/bias:0': 'Online/conv2d_2/bias', 
                'Online/conv2d_2/kernel:0': 'Online/conv2d_2/kernel', 
                'Online/dense/bias:0': 'Online/dense/bias', 
                'Online/dense/kernel:0': 'Online/dense/kernel', 
                'Online/dense_1/bias:0': 'Online/dense_1/bias', 
                'Online/dense_1/kernel:0': 'Online/dense_1/kernel'
            }
        elif self.params['init_from'] in ['dopamine']:        
            vars_mapping = {  
                'Online/conv2d/bias:0': 'Online/Conv/biases', 
                'Online/conv2d/kernel:0': 'Online/Conv/weights', 
                'Online/conv2d_1/bias:0': 'Online/Conv_1/biases', 
                'Online/conv2d_1/kernel:0': 'Online/Conv_1/weights', 
                'Online/conv2d_2/bias:0': 'Online/Conv_2/biases', 
                'Online/conv2d_2/kernel:0': 'Online/Conv_2/weights', 
                'Online/dense/bias:0': 'Online/fully_connected/biases', 
                'Online/dense/kernel:0': 'Online/fully_connected/weights', 
                'Online/dense_1/bias:0': 'Online/fully_connected_1/biases', 
                'Online/dense_1/kernel:0': 'Online/fully_connected_1/weights'
            }
        else:
            print('Source network format unknown!')
            sys.exit()
        # Read checkpoint from file
        checkpoint = NewCheckpointReader(self.params['init_weights'])
        # Get the actual variables from the graph
        trainables_online = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES,
            scope='Online')
        # Copy weights from checkpoint to online network
        for variable in trainables_online:
            #print(variable.name)
            #values = self.sess.run(variable.name)
            #print('Original:', values)
            #saved_values = checkpoint.get_tensor(vars_mapping[variable.name])
            #print('Saved:', saved_values)
            self.sess.run(
                variable.assign(
                    checkpoint.get_tensor(vars_mapping[variable.name])
                ))
            #new_values = self.sess.run(variable.name)
            #print('New:', new_values)


class DQNAgent(Agent):
    """ Implementation of the classic DQN agent. """
    def __init__(
            self,
            params, 
            private_params, 
            private_paths,
            sess,
            summary_writer=None):
        super(DQNAgent, self).__init__(
            params=params, 
            private_params=private_params, 
            private_paths=private_paths, 
            sess=sess,
            summary_writer=summary_writer)
        self.name = 'DQNAgent'

    def build_networks(self):
        """Builds the online and target networks from the network
        template defined in the agent parameters.
        """
        # Prepare the online network which is consistently updated
        self.online_net = tf.make_template(
            'Online',
            self.model_template,
            name='q_values',
            output_shape=self.private_params['action_count'], 
            trainable=True)
        # Q-VALUES (for acting)
        self.online = self.online_net(self.state_ph)
        self.variable_summaries(self.online, 'Online')
        # ACTION ID (for acting)
        self.online_argmax = tf.argmax(
            self.online.q_values, axis=1)[0]  
        # BATCH Q-VALUES (for training)
        self.online_batch = self.online_net(self.memory.states)

        # Prepare the target network that is only updated periodically
        self.target_net = tf.make_template(
            'Target',
            self.model_template,
            name='q_values',
            output_shape=self.private_params['action_count'], 
            trainable=False)
        # BATCH Q'-VALUES (for training)
        self.target_batch = self.target_net(self.memory.next_states)

    def build_train_op(self):
        """Builds the training operation.

        Returns
        -------
        train_op : function
            Function that performs one step of training with batch
            data from replay memory.
        """
        # Draws a number of actions from replay memory and
        # transforms to one hot vector
        memory_batch_action_one_hot = tf.one_hot(
            self.memory.actions,
            self.private_params['action_count'],
            1., 0.,
            name='MemoryBatchActionOneHot')
        # Transforms the one hot vector to present the actual
        # Q values of that action if run through the online network
        online_batch_q = tf.reduce_sum(
            self.online_batch.q_values * memory_batch_action_one_hot,
            reduction_indices=1,
            name='OnlineBatchQ')
        # Build the training target that is treated as a constant
        train_target = tf.stop_gradient(self.build_train_target())
        # Define the loss function for the agent
        loss = tf.losses.huber_loss(
            labels=train_target,
            predictions=online_batch_q,
            reduction=tf.losses.Reduction.NONE)
        if self.summary_writer is not None:
            with tf.variable_scope('Losses'):
                tf.summary.scalar(
                    'OnlineBatchLoss', tf.reduce_mean(loss))
        return self.optimizer.minimize(tf.reduce_mean(loss))

    def build_train_target(self):
        """Build an operation that calculates the target for the Q-value
        for the currently selected batch.

        Returns
        -------
            target_target : function
                Function that calculats the target Q value.
        """
        # Get the maximum Q-value across the actions dimension from
        # the target network
        next_q_max = tf.reduce_max(
            self.target_batch.q_values,
            1,
            name='NextQMax')
        activation = (1. - tf.cast(self.memory.terminals, tf.float32))
        return tf.add(
            self.memory.rewards,
            self.private_params['cumulative_gamma'] * next_q_max * activation,
            name='TrainTarget')

    def build_sync_ops(self):
        """Builds ops for assigning weights from online to target network.

        Returns
        -------
            ops: list of functions
                List of functions that assigns weights from online
                to target network
        """
        # Get trainable variables from online network
        trainables_online = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES,
            scope='Online')
        # Get global variables from target network
        trainables_target = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES,
            scope='Target')
        # make sure that variables are the same amount in online
        # and target networks
        assert len(trainables_online) == len(trainables_target)
        # Assign weights from online to target network.
        sync_ops = []
        for (w_online, w_target) in zip(trainables_online, trainables_target):
            sync_ops.append(w_target.assign(w_online, use_locking=True))
        return sync_ops

    def select_action(self, episode_type):
        """Select an action from following a given policy.

        Parameters
        ----------
            episode_type : str
                Type of episode, either `train` or `eval`.

        Returns
        -------
            action : int
                Action ID of the selected action.
        """
        '''
        if episode_type in ['render', 'test']:
            # Choose the action with highest Q-value at the current state.
            return self.sess.run(
                self.online_argmax, 
                {self.state_ph: self.state})
        '''
        #TODO: Update to use policy module
        epsilon = self.params['epsilon_eval'] if episode_type in ['eval', 'render', 'test'] else self.get_epsilon(
            self.params['decay_steps'],
            self.private_params['train_step'],
            self.params['min_memory_size'],
            self.params['epsilon_min'])
        if random.random() <= epsilon:
            # Choose a random action with probability epsilon.
            return random.randint(
                0, self.private_params['action_count'] - 1)
        else:
            # Choose the action with highest Q-value at the current state.
            return self.sess.run(
                self.online_argmax, 
                {self.state_ph: self.state})


class A2TAgent(DQNAgent):
    """An implementation of the A2T agent. """

    def __init__(
            self,
            params, 
            private_params, 
            private_paths,
            sess,
            summary_writer=None):
        '''
        super(A2TAgent, self).__init__(
            params=params, 
            private_params=private_params, 
            private_paths=private_paths,
            sess=sess,
            summary_writer=summary_writer)
        '''
        self.name = 'A2TAgent'
        self.params = params
        self.private_params = private_params
        self.private_params['train_step'] = 0
        self.private_params['agent_step'] = 0
        self.private_params['target_updates'] = 0
        self.private_paths = private_paths

        self.get_epsilon = linearly_decaying_factor
        self.softmax_temp = self.params['softmax_temp']
        self.get_softmax_temp = linearly_decaying_factor

        self.sess = sess
        self.summary_writer=summary_writer
        # Initialize the agent
        with tf.device(self.private_params['device']):
            # build the replay memory
            self.memory = prepare_memory(self.params, self.private_params)
            # build the optimzer
            self.optimizer = prepare_optimizer(self.params)
            # Build the model template
            self.model_template = prepare_model_template(self.params)
            #TODO: Build agent policy
            #self.policy = prepare_policy(self.params, self.private_params)
            # Define placeholder for the network input 
            self.state_ph = tf.placeholder(
                shape=self.private_params['state_shape'],
                dtype=self.private_params['observation_dtype'],
                name='state_ph')
            # Define placeholder for the softmax variable
            self.softmax_ph = tf.placeholder(
                dtype=tf.float32,
                shape=(),
                name='softmax_ph') 
            # Make a dummy state representation
            self.state = np.zeros(self.private_params['state_shape'])
            # Actually build the networks
            self.build_networks()
            # Define the training operations for the network
            self.train_op = self.build_train_op()  # pylint: disable=E1111
            # Define the sync operations between networks
            self.sync_ops = self.build_sync_ops()  # pylint: disable=E1111
        # Combine all summaries
        if self.summary_writer is not None:
            self.merged_summaries = tf.summary.merge_all()
        # TODO: Load pretrained agents in network
        #self.load_pretrained_weights()
        #if self.params['init_weights'] is not None:
        #    self.init_weights()
        # Inititialize the checkpoint saver and limit checkpoints to
        # to save memory
        if self.params['mode'] is not 'render':
            self.checkpoint_saver = tf.train.Saver(
                max_to_keep=self.params['checkpoints_keep_max'])
        # Prepare variables that the agent needs to keep track on
        self.observation = None
        self.last_observation = None

    def build_networks(self):
        """Builds the online and target networks from the network
        template defined in the agent parameters.
        """
        #############################################
        ######## AGENT THAT WE WANT TO TRAIN ########
        #############################################
        # Prepare the online network which is consistently updated
        self.online_net = tf.make_template(
            'Online',
            self.model_template,
            name='q_values',
            output_shape=self.private_params['action_count'], 
            trainable=True)
        # Q-VALUES (for acting)
        self.online = self.online_net(self.state_ph)
        self.variable_summaries(self.online, 'Online')
        # ACTION ID (for acting)
        self.online_argmax = tf.argmax(
            self.online.q_values, axis=1)[0]  
        # BATCH Q-VALUES (for training)
        self.online_batch = self.online_net(self.memory.states)

        # Prepare the target network that is only updated periodically
        self.target_net = tf.make_template(
            'Target',
            self.model_template,
            name='q_values',
            output_shape=self.private_params['action_count'], 
            trainable=False)
        # BATCH Q_PRIME-VALUES (for training)
        self.target_batch = self.target_net(self.memory.next_states)

        #############################################
        ############# PRETRAINED AGENTS #############
        #############################################
        # preapring a dictionary to collect all pretrained agents
        self.pretrained_net = collections.OrderedDict()
        self.pretrained_online = collections.OrderedDict()
        self.pretrained_online_batch = collections.OrderedDict()
        self.pretrained_target_batch = collections.OrderedDict()
        # Iterate over all pretrained agents
        for i in range(len(self.params['pretrained_agents'])):
            # Preparing the network template for each pretrained agent
            self.pretrained_net[i] = tf.make_template(
                'Pretrained{}'.format(i),
                self.model_template,
                name='q_values',
                output_shape=self.private_params['action_count'],
                trainable=False)
            # Q-VALUES for pretrained agents (for acting)
            self.pretrained_online[i] = self.pretrained_net[i](self.state_ph)
            self.variable_summaries(
                self.pretrained_online[i], 
                'Pretrained{}'.format(i))
            # BATCH Q-VALUES for pretrained agents (for training)
            self.pretrained_online_batch[i] = self.pretrained_net[i](
                self.memory.states)
            # BATCH Q_PRIME-VALUES for pretrained agents (for training)
            self.pretrained_target_batch[i] = self.pretrained_net[i](
                self.memory.next_states)

        #############################################
        ############ ATTENTION NETWORK ##############
        #############################################
        # Preparing network template for online attention network
        self.online_weights_net = tf.make_template(
            'OnWeights',
            self.model_template,
            name='weights',
            output_shape=len(self.params['pretrained_agents'])+1, 
            trainable=True)
        # SOFTMAX WEIGHTS (for acting)
        self.online_weights = self.online_weights_net(self.state_ph)
        self.online_weights_soft = tf.nn.softmax(
            (self.online_weights.weights / tf.reduce_max(self.online_weights.weights)) / 
            self.softmax_ph)
        self.variable_summaries(
            self.online_weights_soft,
            'OnWeightsSoft')
        # Create summaries for each softmax network output node
        for i in range(len(self.params['pretrained_agents']) + 1):
            self.variable_summaries(
                self.online_weights_soft[0, i], 
                'OnWeightsSoft_{}'.format(i))
        # BATCH SOFTMAX WEIGHTS (for training)
        self.online_batch_weights = self.online_weights_net(
            self.memory.states)
        self.online_batch_weights_soft = tf.nn.softmax(
            (self.online_batch_weights.weights / tf.reduce_max(self.online_batch_weights.weights)) /
            self.softmax_ph)
        # Preparing network template for target attention network
        self.target_weights_temp = tf.make_template(
            'TarWeights',
            self.model_template,
            name='weights',
            output_shape=len(self.params['pretrained_agents'])+1,
            trainable=False)
        # BATCH SOFTMAX WEIGHTS PRIME (for training)
        self.target_batch_weights = self.target_weights_temp(
            self.memory.next_states)
        self.target_batch_weights_soft = tf.nn.softmax(
            (self.target_batch_weights.weights / tf.reduce_max(self.target_batch_weights.weights)) /
            self.softmax_ph)

        #############################################
        ############## NETWORK STACKS ###############
        #############################################
        # ONLINE Q-VALUES - ACTION SELECTION
        online_q_values = []
        online_q_values.append(self.online.q_values)
        for i in range(len(self.params['pretrained_agents'])):
            online_q_values.append(self.pretrained_online[i].q_values)
        online_stacked_q_values = tf.stack(
            online_q_values, 
            axis=1,
            name='online_stacked_q_values')
        # ONLINE Q-VALUES - BATCH TRAINING
        online_batch_q_values = []
        online_batch_q_values.append(self.online_batch.q_values)
        for i in range(len(self.params['pretrained_agents'])):
            online_batch_q_values.append(self.pretrained_online_batch[i].q_values)
        online_batch_stacked_q_values = tf.stack(
            online_batch_q_values, 
            axis=1, 
            name='online_batch_stacked_q_values')
        # TARGET Q_PRIME-VALUE - BATCH TRAINING
        target_batch_q_values = []
        target_batch_q_values.append(self.target_batch.q_values)
        for i in range(len(self.params['pretrained_agents'])):
            target_batch_q_values.append(self.pretrained_target_batch[i].q_values)
        target_batch_stacked_q_values = tf.stack(
            target_batch_q_values, 
            axis=1, 
            name='target_batch_stacked_q_values')
        
        #############################################
        ########### FINAL NETWORK OUTPUT ############
        #############################################
        # ONLINE Q-VALUES - ACTION SELECTION
        self.online_final = tf.einsum(
            'ij,ijk->ik',
            self.online_weights_soft,
            online_stacked_q_values,
            name='online_q_values')
        self.variable_summaries(
            self.online_final,
            'OnlineFinal')
        # ACTION (for acting)
        self.online_final_argmax = tf.argmax(self.online_final, axis=1)[0]
        # ONLINE Q-VALUES - BATCH TRAINING
        self.online_batch_final = tf.einsum(
            'ij,ijk->ik',
            self.online_batch_weights_soft,
            online_batch_stacked_q_values,
            name='online_batch_q_values')
        # TARGET Q-VALUE - BATCH TRAINING
        self.target_batch_final = tf.einsum(
            'ij,ijk->ik',
            self.target_batch_weights_soft,
            target_batch_stacked_q_values,
            name='target_batch_q_values')

    def build_train_op(self):
        """Builds the training operation.

        Returns
        -------
        train_op : function
            Function that performs one step of training with batch
            data from replay memory.
        """
        # Draws a number of actions from replay memory and
        # transforms to one hot vector
        memory_batch_action_one_hot = tf.one_hot(
            self.memory.actions,
            self.private_params['action_count'],
            1., 0.,
            name='MemoryBatchActionOneHot')
        # Transforms the one hot vector to present the actual
        # Q values of that action if run through the online network
        online_batch_q = tf.reduce_sum(
            self.online_batch.q_values * memory_batch_action_one_hot,
            reduction_indices=1,
            name='OnlineBatchQ')
        # Transforms the one hot vector to present the actual
        # Q values of that action if run through the final online network
        online_batch_final_q = tf.reduce_sum(
            self.online_batch_final * memory_batch_action_one_hot,
            reduction_indices=1,
            name='OnlineBatchFinalQ')    
        # Build the training target that is treated as a constant
        train_target = tf.stop_gradient(self.build_train_target())
        # Prepare the variables that are updated during training
        var_list = tf.trainable_variables()
        var_list_agent = [
            v for v in var_list if v.name.startswith("Online")]
        var_list_weight = [
            v for v in var_list if v.name.startswith("OnWeights")]
        # Define the loss function for the agent
        online_batch_loss = tf.losses.huber_loss(
            labels=train_target,
            predictions=online_batch_q,
            reduction=tf.losses.Reduction.NONE)
        # Define the loss function for the attention network
        online_batch_final_loss = tf.losses.huber_loss(
            labels=train_target,
            predictions=online_batch_final_q,
            reduction=tf.losses.Reduction.NONE)
        # Add loss to summary writer
        if self.summary_writer is not None:
            with tf.variable_scope('Losses'):
                tf.summary.scalar(
                    'OnlineBatchLoss',
                    tf.reduce_mean(online_batch_loss))
                tf.summary.scalar(
                    'OnlineBatchFinalLoss', 
                    tf.reduce_mean(online_batch_final_loss)) 
        # Prepare actual network training functions   
        online_batch_train = self.optimizer.minimize(
            tf.reduce_mean(online_batch_loss),
            var_list=var_list_agent)
        online_batch_final_train = self.optimizer.minimize(
            tf.reduce_mean(online_batch_final_loss),
            var_list=var_list_weight)
        return tf.group(online_batch_train, online_batch_final_train)

    def build_train_target(self):
        """Build an operation that calculates the target for the Q-value
        for the currently selected batch.

        Returns
        -------
            target_target : function
                Function that calculats the target Q value.
        """
        # Get the maximum Q-value across the actions dimension from
        # the target network
        next_q_max = tf.reduce_max(
            self.target_batch_final,
            1,
            name='NextQMax')
        activation = (1. - tf.cast(self.memory.terminals, tf.float32))
        return tf.add(
            self.memory.rewards,
            self.private_params['cumulative_gamma'] * next_q_max * activation,
            name='TrainTarget')

    def build_sync_ops(self):
        """Builds ops for assigning weights from online to target network.

        Returns
        -------
            ops: list of functions
                List of functions that assigns weights from online
                to target network
        """
        # Get trainable variables from online agent network
        trainables_online = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES,
            scope='Online')
        # Get global variables from target agent network
        trainables_target = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES,
            scope='Target')
        # Make sure that variables are the same amount in online
        # and target networks
        assert len(trainables_online) == len(trainables_target)
        # Assign weights from online to target network.
        sync_ops = []
        for (w_online, w_target) in zip(trainables_online, trainables_target):
            sync_ops.append(w_target.assign(w_online, use_locking=True))
        # Get trainable variables from online attention network
        trainables_online = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES,
            scope='OnWeights')
        # Get global variables from target attention network
        trainables_target = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES,
            scope='TarWeights')
        # Make sure that variables are the same amount in online
        # and target networks
        assert len(trainables_online) == len(trainables_target)
        # Assign weights from online to target network.
        for (w_online, w_target) in zip(trainables_online, trainables_target):
            sync_ops.append(w_target.assign(w_online, use_locking=True))
        return sync_ops

    def perform_train_step(self):
        """Runs a single training step.

        Runs a training op if both:
        (1) A minimum number of frames have been added to the replay buffer.
        (2) `training_step` is a multiple of `update_period`.

        Also, syncs weights from online to target network if training steps is a
        multiple of target update period.
        """
        # Run a train op at the rate of self.update_period if enough training steps
        # have been run. This matches the Nature DQN behaviour.
        if self.memory.memory.add_count > self.params['min_memory_size']:
            # TODO: Check if we can solve this better. Now the steps are taken inside the
            # atari environment preprocessor and not actually by the agent
            if self.private_params['agent_step'] % self.params['train_frequency'] == 0:
                self.sess.run(self.train_op, {self.softmax_ph: self.softmax_temp})
                self.private_params['train_step'] += 1
                if (self.private_params['train_step'] % self.params['summary_frequency'] == 0):
                    summary = self.sess.run(
                        self.merged_summaries,
                        {self.state_ph: self.state,
                         self.softmax_ph: self.softmax_temp})
                    self.summary_writer.add_summary(summary, self.private_params['train_step'])
                    self.summary_writer.flush()
                    
                if self.private_params['train_step'] % self.params['target_update_frequency'] == 0:
                    #tf.logging.info('\t{} - Updating target networks: Step {}'.format(self.name, self.private_params['train_step']))
                    self.sess.run(self.sync_ops)
                    self.private_params['target_updates'] += 1

    def select_action(self, episode_type):
        """Select an action from following a given policy.

        Parameters
        ----------
            episode_type : str
                Type of episode, either `train` or `eval`.

        Returns
        -------
            action : int
                Action ID of the selected action.
        """
        # Get probability of taking a random action
        epsilon = self.params['epsilon_eval'] if episode_type in ['eval', 'render', 'test'] else self.params['epsilon_min']
        #    self.get_epsilon(
        #        self.params['decay_steps'],
        #        self.private_params['train_step'],
        #        self.params['min_memory_size'],
        #        self.params['epsilon_min'])
        if random.random() <= epsilon:
            # Choose a random action with probability epsilon.
            return random.randint(
                0, self.private_params['action_count'] - 1)
        else:
            # In evaluation or rendering get the action from the target network
            if episode_type in ['eval', 'render']: 
                return self.sess.run(
                    self.online_argmax,
                    {self.state_ph: self.state})
            # In training get the action from the whole network
            elif episode_type in ['train']:
                self.softmax_temp = self.get_softmax_temp(
                    self.params['softmax_decay_steps'],
                    self.private_params['train_step'],
                    self.params['min_memory_size'],
                    self.params['softmax_min']
                )
                return self.sess.run(
                    self.online_final_argmax,
                    {self.state_ph: self.state, 
                     self.softmax_ph: self.softmax_temp})
            else:
                print('AgentActionSelection: Not sure what we are supposed to do...')
                sys.exit()

    def load_pretrained_agents(self):
        """ Loads the weights of a pretrained agent into the network. """
        # Initialize list of syncronization operations
        #sync_ops = []
        for i in range(len(self.params['pretrained_agents'])):
            # Get all variables that we want to set in the target network
            vars_random = tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES,
                scope='Pretrained{}'.format(i))
            
            # Get a variable mapping for different network names
            if self.params['pretrained_from'][i] in ['RL2go']:
                #vars_mapping = {}
                #for var in vars_random:
                #    tmp = var.name.replace('Pretrained','').split('/')
                #    value = 'Online/{}'.format('/'.join(tmp[1:]))
                #    vars_mapping[var.name] = value
                vars_mapping = {  
                    'Pretrained{}/conv2d/bias:0'.format(i): 'Online/conv2d/bias', 
                    'Pretrained{}/conv2d/kernel:0'.format(i): 'Online/conv2d/kernel', 
                    'Pretrained{}/conv2d_1/bias:0'.format(i): 'Online/conv2d_1/bias', 
                    'Pretrained{}/conv2d_1/kernel:0'.format(i): 'Online/conv2d_1/kernel', 
                    'Pretrained{}/conv2d_2/bias:0'.format(i): 'Online/conv2d_2/bias', 
                    'Pretrained{}/conv2d_2/kernel:0'.format(i): 'Online/conv2d_2/kernel', 
                    'Pretrained{}/dense/bias:0'.format(i): 'Online/dense/bias', 
                    'Pretrained{}/dense/kernel:0'.format(i): 'Online/dense/kernel', 
                    'Pretrained{}/dense_1/bias:0'.format(i): 'Online/dense_1/bias', 
                    'Pretrained{}/dense_1/kernel:0'.format(i): 'Online/dense_1/kernel'
                }
            elif self.params['pretrained_from'][i] in ['dopamine']:
                vars_mapping = {
                    'Pretrained{}/conv2d/bias:0'.format(i): 'Online/Conv/biases', 
                    'Pretrained{}/conv2d/kernel:0'.format(i): 'Online/Conv/weights', 
                    'Pretrained{}/conv2d_1/bias:0'.format(i): 'Online/Conv_1/biases', 
                    'Pretrained{}/conv2d_1/kernel:0'.format(i): 'Online/Conv_1/weights', 
                    'Pretrained{}/conv2d_2/bias:0'.format(i): 'Online/Conv_2/biases', 
                    'Pretrained{}/conv2d_2/kernel:0'.format(i): 'Online/Conv_2/weights', 
                    'Pretrained{}/dense/bias:0'.format(i): 'Online/fully_connected/biases', 
                    'Pretrained{}/dense/kernel:0'.format(i): 'Online/fully_connected/weights', 
                    'Pretrained{}/dense_1/bias:0'.format(i): 'Online/fully_connected_1/biases', 
                    'Pretrained{}/dense_1/kernel:0'.format(i): 'Online/fully_connected_1/weights'
                }
            else:
                print('Source network format unknown: {}!'.format(self.params['pretrained_from'][i]))
                sys.exit()
            
            # Read checkpoint from file
            checkpoint = NewCheckpointReader(self.params['pretrained_agents'][i])
            # Get the actual variables from the graph
            pretrained_variables = tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES,
                scope='Pretrained{}'.format(i))
            # Copy weights from checkpoint to online network
            for variable in pretrained_variables:
                self.sess.run(
                    variable.assign(
                        checkpoint.get_tensor(vars_mapping[variable.name])
                    ))

        '''
            # Add syncronization operations to assign weights from source to target network
            for vr in vars_random:
                sync_ops.append(
                    vr.assign(
                        tf.contrib.framework.load_variable(
                            self.params['pretrained_agents'][i],
                            vars_mapping[vr.name]),
                        use_locking=True))
        # Run all syncronization operations
        self.sess.run(sync_ops)
        '''



class BaselineAgent(DQNAgent):
    """An implementation of a baseline agent. """

    def __init__(
            self,
            params, 
            private_params, 
            private_paths,
            sess,
            summary_writer=None):
        super(BaselineAgent, self).__init__(
            params=params, 
            private_params=private_params, 
            private_paths=private_paths,
            sess=sess,
            summary_writer=summary_writer)
        self.name = 'BaselineAgent'

    def build_networks(self):
        """Builds the online and target networks from the network
        template defined in the agent parameters.
        """
        # Prepare the online network which is consistently updated
        self.online_net = tf.make_template(
            'Online',
            self.model_template,
            name='q_values',
            output_shape=self.private_params['action_count'], 
            trainable=True)
        # Q-VALUES (for acting)
        self.online = self.online_net(self.state_ph)
        self.variable_summaries(self.online, 'Online')
        # ACTION ID (for acting)
        self.online_argmax = tf.argmax(
            self.online.q_values, axis=1)[0]  
        # BATCH Q-VALUES (for training)
        self.online_batch = self.online_net(self.memory.states)

        # Prepare the target network that is only updated periodically
        self.target_net = tf.make_template(
            'Target',
            self.model_template,
            name='q_values',
            output_shape=self.private_params['action_count'], 
            trainable=False)
        # BATCH Q'-VALUES (for training)
        self.target_batch = self.target_net(self.memory.next_states)

        # Prepare the pretrained network 
        self.pretrained_net = tf.make_template(
            'Pretrained',
            self.model_template,
            name='q_values',
            output_shape=self.private_params['action_count'], 
            trainable=True)
        # Q-VALUES (for acting)
        self.pretrained = self.pretrained_net(self.state_ph)
        self.variable_summaries(self.pretrained, 'Pretrained')
        # ACTION ID (for acting)
        self.pretrained_argmax = tf.argmax(
            self.pretrained.q_values, axis=1)[0] 
        # BATCH Q'-VALUES (for training)
        self.pretrained_batch = self.pretrained_net(self.memory.next_states)

    def build_train_target(self):
        """Build an operation that calculates the target for the Q-value
        for the currently selected batch.

        Returns
        -------
            target_target : function
                Function that calculats the target Q value.
        """
        # Get the maximum Q-value across the actions dimension from
        # the target network
        if self.params['target_net'] in ['target']:
            next_q_max = tf.reduce_max(
                self.target_batch.q_values,
                1,
                name='NextQMax')
        elif self.params['target_net'] in ['pretrained']:
            next_q_max = tf.reduce_max(
                self.pretrained_batch.q_values,
                1,
                name='NextQMax')
        else:
            print('BuildTrainTarget: Wrong source for training target')
            sys.exit()
        activation = (1. - tf.cast(self.memory.terminals, tf.float32))
        return tf.add(
            self.memory.rewards,
            self.private_params['cumulative_gamma'] * next_q_max * activation,
            name='TrainTarget')

    def select_action(self, episode_type):
        """Select an action from following a given policy.

        Parameters
        ----------
            episode_type : str
                Type of episode, either `train` or `eval`.

        Returns
        -------
            action : int
                Action ID of the selected action.
        """
        # Get probability of taking a random action
        epsilon = self.params['epsilon_eval'] if episode_type in ['eval', 'test'] else self.params['epsilon_min']
        if random.random() <= epsilon:
            # Choose a random action with probability epsilon.
            return random.randint(
                0, self.private_params['action_count'] - 1)
        else:
            # In evaluation get the action from the target network
            if episode_type in ['eval', 'render']: 
                return self.sess.run(
                    self.online_argmax, 
                    {self.state_ph: self.state})
            # In training get the action from the pretrained network
            elif episode_type in ['train']:
                return self.sess.run(
                    self.pretrained_argmax,
                    {self.state_ph: self.state})
            else:
                print('AgentActionSelection: Not sure what we are supposed to do...')
                sys.exit()
        
    def load_pretrained_agents(self):
        """ Loads the weights of a pretrained agent into the network. """
        # Define variable mapping according to agent source
        if self.params['pretrained_from'][0] in ['RL2go']:
            vars_mapping = {  
                'Pretrained/conv2d/bias:0': 'Online/conv2d/bias', 
                'Pretrained/conv2d/kernel:0': 'Online/conv2d/kernel', 
                'Pretrained/conv2d_1/bias:0': 'Online/conv2d_1/bias', 
                'Pretrained/conv2d_1/kernel:0': 'Online/conv2d_1/kernel', 
                'Pretrained/conv2d_2/bias:0': 'Online/conv2d_2/bias', 
                'Pretrained/conv2d_2/kernel:0': 'Online/conv2d_2/kernel', 
                'Pretrained/dense/bias:0': 'Online/dense/bias', 
                'Pretrained/dense/kernel:0': 'Online/dense/kernel', 
                'Pretrained/dense_1/bias:0': 'Online/dense_1/bias', 
                'Pretrained/dense_1/kernel:0': 'Online/dense_1/kernel'
            }
        elif self.params['pretrained_from'][0] in ['dopamine']:        
            vars_mapping = {  
                'Pretrained/conv2d/bias:0': 'Online/Conv/biases', 
                'Pretrained/conv2d/kernel:0': 'Online/Conv/weights', 
                'Pretrained/conv2d_1/bias:0': 'Online/Conv_1/biases', 
                'Pretrained/conv2d_1/kernel:0': 'Online/Conv_1/weights', 
                'Pretrained/conv2d_2/bias:0': 'Online/Conv_2/biases', 
                'Pretrained/conv2d_2/kernel:0': 'Online/Conv_2/weights', 
                'Pretrained/dense/bias:0': 'Online/fully_connected/biases', 
                'Pretrained/dense/kernel:0': 'Online/fully_connected/weights', 
                'Pretrained/dense_1/bias:0': 'Online/fully_connected_1/biases', 
                'Pretrained/dense_1/kernel:0': 'Online/fully_connected_1/weights'
            }
        else:
            print('Source network format unknown: {}!'.format(self.params['pretrained_from']))
            sys.exit()
        # Read checkpoint from file
        checkpoint = NewCheckpointReader(self.params['pretrained_agents'][0])
        # Get the actual variables from the graph
        pretrained_variables = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES,
            scope='Pretrained')
        # Copy weights from checkpoint to online network
        for variable in pretrained_variables:
            #print(variable.name)
            #values = self.sess.run(variable.name)
            #print('Original:', values)
            #saved_values = checkpoint.get_tensor(vars_mapping[variable.name])
            #print('Saved:', saved_values)
            self.sess.run(
                variable.assign(
                    checkpoint.get_tensor(vars_mapping[variable.name])
                ))
            #new_values = self.sess.run(variable.name)
            #print('New:', new_values)

    