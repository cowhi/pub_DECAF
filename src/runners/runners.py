import math
import os
import random
import sys
import time
from copy import deepcopy
from pathlib import Path

import yaml
import numpy as np
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.INFO)

sys.path.append('..')
import util 
from agents import (  # pylint: disable=import-error
    Checkpointer,
    get_latest_checkpoint_number, 
    Logger, 
    prepare_agent,
    Statistics
)
from envs import prepare_env  # pylint: disable=import-error


class Runner(object):

    def __init__(self, run, params):
        """Initialization of runner class which coordinates one run
        of an agent in a given environment under the given parameters.

        Parameters
        ----------
        run : int
            Number of run in the current experiment.
        params : dict
            Directory that contains all experiment settings.
        """
        self.name = 'Runner'
        tf.logging.info('\t{} - Creating experiment runner ...'.format(self.name))
        # Initialize important parameter
        self.params = params
        self.private_params, self.private_paths = self.make_private_params(run)
        self.set_random_seeds()
        # Prepare the environment properly
        self.env = prepare_env(self.params, self.private_params['seed'])
        self.private_params['observation_shape'] = self.env.observation_space.shape
        self.private_params['observation_dtype'] = self.env.observation_space.dtype
        
        if len(self.private_params['observation_shape']) == 3:
            self.private_params['observation_shape'] = self.private_params['observation_shape'][:-1]
        self.private_params['state_shape'] = (1,) \
                                            + self.private_params['observation_shape'] \
                                            + (self.params['stack_size'],)
        self.private_params['action_count'] = self.env.action_space.n
        # Prepare a simple logger
        if self.params['mode'] is not 'render':
            self.logger = Logger(
                self.private_paths['log_dir'],
                self.params['checkpoints_keep_max']
                )
        # Start a tensorflow session and add a summary_writer
        tf.reset_default_graph()
        self.sess = tf.Session(
            '', config=self.params['tf_config'])
        if self.params['mode'] is not 'render':
            self.summary_writer = tf.summary.FileWriter(
                self.private_paths['base_dir'])
        else:
            self.summary_writer = None
        
        # Prepare the agent and save a summary
        self.agent = prepare_agent(
            self.params, 
            self.private_params,
            self.private_paths,
            self.sess,
            self.summary_writer)
        if self.params['mode'] is not 'render':
            self.summary_writer.add_graph(graph=tf.get_default_graph())
            self.summary_writer.flush()
            self.summary_writer.close()
            # Open the summary writer again and initialize the session
            self.summary_writer.reopen()
        #else:
        #    print('Reload not implemented yet!')
        self.sess.run(
            tf.global_variables_initializer())
        # Load pretrained weights if available
        if self.params['init_weights']:
            self.agent.init_weights()
        if self.params['pretrained_agents'] is not None:
            self.agent.load_pretrained_agents()
        # Check if session needs to be resumed or starts from scratch
        self.init_checkpoint()

    def make_private_params(self, run):
        """Initializes many experiment specific variables.
        
        Parameters
        ----------
            run : int
                Number of the current experiment.

        Returns
        -------
            private_params : dict
                Directory that contains all experiment specific settings.
        """
        private_params = {
            'run' : run,
            'seed' : self.params['random_seed'][run - 1],
            'device' : '/gpu:*' if self.params['use_gpu'] else '/cpu:*',
            'start_era' : 0,
            'current_era' : 0,
            'current_episode' : 0,
            'current_step' : 0,
            'eval_episodes' : 0,
            'eval_steps' : 0,
            'cumulative_gamma' : math.pow(self.params['gamma'], self.params['update_horizon'])
        }
        private_paths = {}
        private_paths['base_dir'] = str(Path(os.path.join(
            self.params['exp_dir'], 
            'run_{:02d}_s{}'.format(
                run,
                private_params['seed']))))
        private_paths['ckpt_dir'] = str(Path(
            os.path.join(private_paths['base_dir'], 'checkpoints')
        ))
        private_paths['log_dir'] = str(Path(
            os.path.join(private_paths['base_dir'], 'logs')
        ))
        private_paths['var_file'] = str(Path(
            os.path.join(private_paths['base_dir'], 'vars.yaml')
        ))
        if self.params['source_dir'] is not None:
            util.copy_dir(self.params['source_dir'], private_paths['base_dir'])  # pylint: disable=no-member
        return private_params, private_paths

    def set_random_seeds(self):
        """Sets all relevant random seeds to improve reproducability."""
        np.random.seed(self.private_params['seed'])
        random.seed(self.private_params['seed'])
        tf.set_random_seed(self.private_params['seed'])

    def get_base_dir(self):
        """Getter for the base directory of the experiment.
        
        Returns
        -------
        base_dir : str
            Base directory of experiment.
        """
        return self.private_paths['base_dir']

    def init_checkpoint(self):
        """Initializes the checkpoint saver and loads experiment
        parameter from checkpoint if necessary.

        Returns
        -------
        start_era : int
            Number of era that will be trained in next.
        """
        self.checkpointer = Checkpointer(
            base_directory=self.private_paths['ckpt_dir'],
            checkpoint_file_prefix=self.params['checkpoint_prefix'],
            checkpoint_frequency=self.params['checkpoint_frequency'],
            checkpoints_keep_max=self.params['checkpoints_keep_max'])
        #start_era = 0
        # Check if checkpoint exists. Note that the existence of checkpoint 0 means
        # that we have finished era 0 (so we will start from era 1).
        latest_checkpoint_number = get_latest_checkpoint_number(
            self.private_paths['ckpt_dir'])
        if latest_checkpoint_number >= 0:
            experiment_data = self.checkpointer.load_checkpoint(
                latest_checkpoint_number)

            if self.agent.unbundle(
                    latest_checkpoint_number,
                    experiment_data):
                if self.params['mode'] not in ['render']:
                    assert 'logs' in experiment_data
                    self.logger.data = experiment_data['logs']
                    assert 'current_era' in experiment_data
                    assert 'private_params' in experiment_data
                    for key in ['current_episode', 'current_step', 'eval_episodes', 'eval_steps', 'train_step', 'agent_step']:
                        self.private_params[key] = experiment_data['private_params'][key]
                    self.private_params['start_era'] = experiment_data['current_era'] + 1
                    self.private_params['current_era'] = self.private_params['start_era']
                
                #print('AGENT_PARAMS {}'.format(experiment_data['logs']))
                #print('RUNNER_PARAMS {}'.format(self.private_params))

    def run_experiment(self):
        self.private_params['start_time'] = time.time()

        # make sure we are not running too many eras
        if self.params['eras'] <= self.private_params['start_era']:
            print('\t{} - eras ({}) < start_era({}))'.format(
                self.name,
                self.params['eras'],
                self.private_params['start_era']))
            return

        # Run a first era to test the untrained agent
        if self.private_params['start_era'] == 0:
            self.private_params['current_era'] = 0
            era_start_time = time.time()
            era_statistics = Statistics()
            # Make dummy entry so there are no errors in visualization later
            train_epoch_log = {
                'episodes': np.nan,
                'steps': np.nan,
                'rewards': np.nan,
                'reward_avg': np.nan,
                'step_avg': np.nan,
                'time': np.nan,
            }
            era_statistics.append({
                'currentEpisode': self.private_params['current_episode'],
                'currentStep': self.private_params['current_step'],
                'trainEpisodeSteps': train_epoch_log['steps'],
                'trainEpisodeRewards': train_epoch_log['rewards'],
                'trainEpisodeTimes': train_epoch_log['time'],
                'trainEpisodeRewardsPerStep': np.nan,
                'trainEpochRewardPerEpisodeAvg': train_epoch_log['reward_avg'],
                'trainEpochStepPerEpisodeAvg': train_epoch_log['step_avg'],
                'trainEpochRewardPerStepAvg': np.nan,
                'trainEpochTimePerEpisodeAvg': np.nan,
                'trainEpochTime': train_epoch_log['time']
            })
            train_time = time.time() - era_start_time
            # Make evaluation of untrained agent
            eval_epoch_log = self.run_epoch(
                stats=era_statistics,
                epoch_type='eval')
            eval_time = time.time() - era_start_time - train_time
            # Update summaries for tensorboard
            self.save_tensorboard_summaries(train_epoch_log, eval_epoch_log)
            # Saves the experiment progress to disk
            self.write_logs(era_statistics.data_lists)
            # Generates a checkpoint for the tensorflow graph
            self.write_checkpoint()
            self.private_params['start_era'] += 1
            save_time = time.time() - era_start_time - train_time - eval_time
            self.write_era_time(train_time, eval_time, save_time)
        
        # Run as many erass as defined
        for era in range(self.private_params['start_era'], self.params['eras'] + 1):
            self.private_params['current_era'] = era
            era_start_time = time.time()
            era_statistics = Statistics()
            # Run a training epoch
            train_epoch_log = self.run_epoch(
                stats=era_statistics,
                epoch_type='train')
            train_time = time.time() - era_start_time
            # Run an evaluation epoch
            eval_epoch_log = self.run_epoch(
                stats=era_statistics,
                epoch_type='eval')
            eval_time = time.time() - era_start_time - train_time
            # Update summaries for tensorboard
            self.save_tensorboard_summaries(train_epoch_log, eval_epoch_log)
            # Saves the experiment progress to disk
            self.write_logs(era_statistics.data_lists)
            # Generates a checkpoint for the tensorflow graph
            self.write_checkpoint()
            save_time = time.time() - era_start_time - train_time - eval_time
            self.write_era_time(train_time, eval_time, save_time)
        # Wrap up run
        # Flush and close the summary writer
        self.summary_writer.flush()
        self.summary_writer.close()

        duration = int(time.time() - self.private_params['start_time'])
        tf.logging.info(
                '\t{} - Total run time: {}'
                .format(
                    self.name, 
                    util.get_readable_time(duration)))  # pylint: disable=no-member

    def write_era_time(self, train_time, eval_time, save_time):
        """Writes an output to follow training progress.

        Parameters
        ----------
            epoch_log : dict
                Dictionary containing epoch statistics.
        """
        progress = (
            (self.private_params['current_era']
             - self.private_params['start_era'])
            / self.params['eras'])
        duration = int(time.time() - self.private_params['start_time'])
        tf.logging.info(
                '\t{} - End of era [Train: {}, Eval: {}, Checkpointing: {}], Estimated time remaining: {}'
                .format(
                    self.name, 
                    util.get_readable_time(train_time),  # pylint: disable=no-member
                    util.get_readable_time(eval_time),  # pylint: disable=no-member
                    util.get_readable_time(save_time),  # pylint: disable=no-member
                    util.get_readable_time(  # pylint: disable=no-member
                        util.estimate_remaining_time(duration, progress)  # pylint: disable=no-member
                    ))) 

    def run_epoch(self, stats, epoch_type):
        epoch_log = {
            'episodes' : 0,
            'steps' : 0,
            'reward' : 0,
            'time' : time.time(),
            'reward_avg' : 0,
            'step_avg' : 0,
            'time_avg' : 0,
            'reward_step' : 0,
            'epoch_type': epoch_type
        }
        #if self.params[epoch_type + '_style'] == 'steps':
        while epoch_log[self.params[epoch_type + '_style']] < self.params[epoch_type + '_interval']:
            # Run episode
            episode_log = self.run_episode(epoch_type)
            # Update stats
            stats.append({
                '{}EpisodeRewards'.format(epoch_type): episode_log['reward'],
                '{}EpisodeSteps'.format(epoch_type): episode_log['steps'],
                '{}EpisodeTimes'.format(epoch_type): episode_log['time'],
                '{}EpisodeRewardsPerStep'.format(epoch_type): episode_log['reward']/episode_log['steps']
            })
            epoch_log['episodes'] += 1
            epoch_log['steps'] += episode_log['steps']
            epoch_log['reward'] += episode_log['reward']
            if epoch_type == 'train':
                self.private_params['current_episode'] += 1
                #TODO: Count and train only until epoch limit! 
                self.private_params['current_step'] += episode_log['steps']
            elif epoch_type == 'eval':
                self.private_params['eval_episodes'] += 1
                self.private_params['eval_steps'] += episode_log['steps']
            # Generate some output for the console
            #self.write_to_console(episode_log)
        # Update stats
        epoch_log['time'] = time.time() - epoch_log['time']
        epoch_log['reward_avg'] = epoch_log['reward'] / epoch_log['episodes'] if epoch_log['episodes'] > 0 else 0.0
        epoch_log['step_avg'] = epoch_log['steps'] / epoch_log['episodes'] if epoch_log['episodes'] > 0 else 0.0
        epoch_log['time_avg'] = epoch_log['time'] / epoch_log['episodes'] if epoch_log['episodes'] > 0 else 0.0
        epoch_log['reward_step'] = epoch_log['reward_avg'] / epoch_log['step_avg'] if epoch_log['step_avg'] > 0 else 0.0
        stats.append({
            '{}EpochRewardPerEpisodeAvg'.format(epoch_type): epoch_log['reward_avg'],
            '{}EpochStepPerEpisodeAvg'.format(epoch_type): epoch_log['step_avg'],
            '{}EpochTimePerEpisodeAvg'.format(epoch_type): epoch_log['time_avg'],
            '{}EpochTime'.format(epoch_type): epoch_log['time'],
            '{}EpochRewardPerStepAvg'.format(epoch_type): epoch_log['reward_step']
        })
        if epoch_type == 'train':
            stats.append({
                'currentEpisode': self.private_params['current_episode'],
                'currentStep': self.private_params['current_step']
            })
        if epoch_type == 'eval':
            stats.append({
                'evalEpisode': self.private_params['eval_episodes'],
                'evalStep': self.private_params['eval_steps']
            })
        # Write epoch results to out console
        self.write_epoch_log(epoch_log)
        return epoch_log

    def write_epoch_log(self, epoch_log):
        """Writes an output to follow training progress.

        Parameters
        ----------
            epoch_log : dict
                Dictionary containing epoch statistics.
        """
        tf.logging.info(
            '\t{} - Era {}/{} Step {} Episode {} {} Episodes: {} Reward/Episode: {:.2f} Steps/Episode: {:.2f} Steps/s: {:.2f}'
            .format(self.name,
                    self.private_params['current_era'],
                    self.params['eras'],
                    self.private_params['current_step'],
                    self.private_params['current_episode'],
                    epoch_log['epoch_type'],
                    epoch_log['episodes'],
                    epoch_log['reward_avg'],
                    epoch_log['step_avg'],
                    epoch_log['steps'] / epoch_log['time']))

    def write_to_console(self, episode_log):
        """Writes a status message to the console to monitor training progress.
        
        Parameters
        ----------
            episode_log : dict
                Dictionary containing relevant log data.
        """
        sys.stdout.write(
            'Run {} *'.format(self.private_params['run']) +
            'Era {:>3}/{} *'.format(self.private_params['current_era'], self.params['eras']) +
            'Episodes {:>4} *'.format(self.private_params['current_episode']) +
            'Steps/Episode {:>4} *'.format(episode_log['steps']) +
            'Reward/Episode {:8.2f} *'.format(episode_log['reward']) +
            'Steps/s {:4} *'.format(int(episode_log['steps']/episode_log['time'])) +
            'Remaing ~{}\r'.format(
                util.get_readable_time(  # pylint: disable=no-member
                    util.estimate_remaining_time(  # pylint: disable=no-member
                        time.time() - self.private_params['start_time'],
                        self.private_params['current_era'] / self.params['eras'])))
        )
        sys.stdout.flush()

    def run_episode(self, episode_type):
        episode_log = {
            'steps' : 0,
            'reward' : 0,
            'time' : time.time(),
        }
        action = self.start_episode(episode_type)
        is_terminal = False
        
        # Keep interacting until we reach a terminal state.
        while True:
            # Perform the current action in the environment
            observation, reward, is_terminal = self.run_one_step(action)
            # Update stats
            episode_log['reward'] += reward
            episode_log['steps'] += 1
            # Perform reward clipping only for neural network training.
            reward = np.clip(reward, -1., 1.)
            # workaround for the Atari preprocessing
            #TODO: CHECK IF THIS IS WORKING WITH NEW ENV WRAPPER
            #if ((hasattr(self.env, 'game_over') and self.env.game_over) or
            #        (not hasattr(self.env, 'game_over') and is_terminal) or
            if (is_terminal or
                    episode_log['steps'] == self.params['max_steps_episode']):
                # Stop the run loop once we reach the true end of episode.
                break
            #elif hasattr(self.env, 'game_over') and is_terminal:
            #    # If we lose a life but the episode is not over, signal an artificial
            #    # end of episode to the agent.
            #    self.agent.end_episode(reward, episode_type)
            #    action = self.agent.start_episode(observation, episode_type)
            else:
                action = self.agent.step(reward, observation, episode_type)

        self.end_episode(reward, episode_type)
        episode_log['time'] = time.time() - episode_log['time']
        return episode_log


    def start_episode(self, episode_type):
        observation = self.env.reset()
        return self.agent.start_episode(observation, episode_type)

    def end_episode(self, reward, episode_type):
        """Finalizes an episode run.

        Parameters
        ----------
            reward: float
                Last reward from the environment.
            episode_type : str
                Type of episode, either `train` or `eval`.

        """
        self.agent.end_episode(reward, episode_type)

    def run_one_step(self, action):
        """Executes a single step in the environment.

        Parameters
        ----------
            action : int
                Action ID to perform in the environment.

        Returns
        -------
            observation : np.array
                Observation returned from envrionemt.
            reward : float
                Reward returned from environmeny.
            is_terminal : bool
                State of the environment.
        """
        observation, reward, is_terminal, _ = self.env.step(action)
        return observation, reward, is_terminal

    def save_tensorboard_summaries(self, train_epoch_log, eval_epoch_log):
        """Save statistics as tensorboard summaries.

        Parameters
        ----------
            train_epoch_log : dict
                Dictionary conatining training epoch results.
            eval_epoch_log : dict
                Dictionary conatining evaluation epoch results.
        """
        summary = tf.Summary(value=[
            tf.Summary.Value(
                tag='Train/EpisodesPerEpoch',
                simple_value=train_epoch_log['episodes']),
            tf.Summary.Value(
                tag='Train/AvgRewardPerEpisode',
                simple_value=train_epoch_log['reward_avg']),
            tf.Summary.Value(
                tag='Train/AvgStepsPerEpisode',
                simple_value=train_epoch_log['step_avg']),
            tf.Summary.Value(
                tag='Train/TimePerEpisode',
                simple_value=train_epoch_log['time']),
            tf.Summary.Value(
                tag='Eval/EpisodesPerEpoch',
                simple_value=eval_epoch_log['episodes']),
            tf.Summary.Value(
                tag='Eval/AvgRewardPerEpoch',
                simple_value=eval_epoch_log['reward_avg']),
            tf.Summary.Value(
                tag='Eval/AvgStepsPerEpoch',
                simple_value=eval_epoch_log['step_avg']),
            tf.Summary.Value(
                tag='Eval/TimePerEpoch',
                simple_value=eval_epoch_log['time']),
        ])
        self.summary_writer.add_summary(summary, self.private_params['current_step'])
        self.summary_writer.flush()

    def write_logs(self, statistics):
        """Records the results of the current iteration.

        Parameters
        ----------
            statistics : dict
                Dictionary containing statistics from Logger object.
        """
        #self.logger['era_{:d}'.format(self.private_params['current_era'])] = statistics
        self.logger[self.private_params['current_era']] = statistics
        if self.private_params['current_era'] % self.params['log_interval'] == 0:
            self.logger.log_to_file(
                self.params['log_prefix'],
                self.private_params['current_era'])

    def write_checkpoint(self):
        """Generates a checkpoint of the experiment data and writes it 
        to disk.
        """
        experiment_data = self.agent.bundle_and_checkpoint(
            self.private_params['current_era'],
            self.private_paths['ckpt_dir'])
        #print('AGENT_PARAMS {}'.format(experiment_data))
        #print('RUNNER_PARAMS {}'.format(self.private_params))
        if experiment_data:
            experiment_data['current_era'] = self.private_params['current_era']
            experiment_data['logs'] = self.logger.data
            self.checkpointer.save_checkpoint(
                self.private_params['current_era'],
                experiment_data)
        #self.dump_important_variables()
    '''
    def dump_important_variables(self):
        """Makes a backup of important variables so in case we have
        to restart training from a checkpoint we start under the same
        conditions.
        """
        data = {
            'current_episode': self.private_params['current_episode'],
            'current_step': self.private_params['current_step'],
            'eval_episodes': self.private_params['eval_episodes'],
            'eval_steps': self.private_params['eval_steps']
        }
        with open(self.private_paths['var_file'], 'w') as outfile:
            yaml.dump(data, outfile, default_flow_style=False)
    '''

    def render_episode(self, episode_type='render'):
        episode_log = {
            'steps' : 0,
            'reward' : 0,
            'time' : time.time(),
        }
        action = self.start_episode(episode_type)
        is_terminal = False
        # Keep interacting until we reach a terminal state.
        while True:
            # Perform the current action in the environment
            observation, reward, is_terminal = self.run_one_step(action)
            self.env.render('human')
            # Update stats
            episode_log['reward'] += reward
            episode_log['steps'] += 1
            # Perform reward clipping only for neural network training.
            reward = np.clip(reward, -1., 1.)
            # workaround for the Atari preprocessing
            if (is_terminal or
                    episode_log['steps'] == self.params['max_steps_episode']):
                # Stop the run loop once we reach the true end of episode.
                break
            else:
                action = self.agent.step(reward, observation, episode_type)
        self.end_episode(reward, episode_type)
        episode_log['time'] = time.time() - episode_log['time']
        print('Episode finished in {} after {} steps with reward {}'.format(
            util.get_readable_time(episode_log['time']),  # pylint: disable=no-member
            episode_log['steps'],
            episode_log['reward']
        ))

        

