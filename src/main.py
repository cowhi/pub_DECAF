
# Standard library imports
import argparse
import os
import random
import shutil
import subprocess
import sys
import time
import multiprocessing as mp
from pathlib import Path
# Third party imports
import yaml
import numpy as np
import tensorflow as tf
# Local application imports
from runners import Runner  # pylint: disable=import-error
from util import (  # pylint: disable=no-name-in-module
    create_dir,
    copy_dir,
    copy_file,
    get_available_gpus,
    load_params, 
    seed_generator)


# Script should to be started directly from ./RL2go directory
BASE_DIR = str(Path(Path.cwd()).parent)
PARAMS_DIR = str(Path(os.path.join(BASE_DIR, 'params')))
LOG_DIR = str(Path(os.path.join(BASE_DIR, 'logs')))
PRETRAINED_DIR = str(Path(os.path.join(BASE_DIR, 'pretrained')))


def render(params):
    """ Starts a test run and renders the output without updating the agent.

        Parameters
        ----------
        params : dict
            Directory that contains all experiment settings.
    """
    # Init Runner
    runner = Runner(run=1, params=params)
    runner.render_episode()
    # delete directory
    print('Removing experiment logs from {}'.format(params['exp_dir']))
    shutil.rmtree(params['exp_dir'])

def test(params):
    """ Starts a test run and renders the output without updating the agent.

        Parameters
        ----------
        params : dict
            Directory that contains all experiment settings.
    """
    # Init Runner
    params['random_seed'] = range(params['test_episodes'])
    rewards = []
    for episode in range(1, params['test_episodes']+1):
        runner = Runner(run=episode, params=params)
        episode_results = runner.run_episode('test')
        rewards.append(episode_results['reward'])
        print('Test: {:3d}, Reward: {:.2f}, Current Avg Reward: {:.2f}'.format(episode, episode_results['reward'], sum(rewards) / episode))
    reward_avg = sum(rewards) / params['test_episodes']
    print()
    print('Avg Reward per episode: {:.2f} (after {:d} test episodes)'.format(reward_avg, params['test_episodes']))
    print('Removing experiment logs from {}'.format(params['exp_dir']))
    shutil.rmtree(params['exp_dir'])

def train(params, run):
    """ Starts training run.

        Parameters
        ----------
        params : dict
            Directory that contains all experiment settings.
        run: int
            Number of training run in experiment.

        Returns
        -------
        base_dir : Path
            Returns base directory path of run when complete.
    """
    # Init Runner
    runner = Runner(run=run, params=params)
    # Run 
    runner.run_experiment()
    return runner.get_base_dir()

def main(params):
    """ Main module to guide the experiment according to its parameters.

        Parameters
        ----------
        params : dict
            Directory that contains all experiment settings.
    """
    if params['mode'] == 'train':
        if (params['visualize_train']
                or params['visualize_test'] 
                or params['runs'] == 1):
            # Force only one process or visualization freezes program
            result_paths = []
            for run in range(1,params['runs']+1):
                result_paths.append(train(params, run))
        else:
            # Run training in multiple processes
            with mp.Pool(processes=params['runs']) as pool:
                results = [pool.apply_async(train, args=(params, run)) \
                    for run in range(1, params['runs']+1)]
                result_paths = [p.get() for p in results]
        # Make report for whole experiment (summarize results)
        # TODO
    elif params['mode'] == 'render':
        render(params)
    elif params['mode'] == 'test':
        test(params)
    else:
        print('The mode "{}" has not yet been implemented.'.format(
            params['mode']))
    


if __name__ == "__main__":
    # Get the time to estimate duration
    start_time = time.time()

    # Read the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--mode', choices=['train', 'test', 'render'], default='train')
    parser.add_argument(
        '--params', type=str, default='params.yaml',
        help='Source for experiment parameters (will be copied to log directory).')
    parser.add_argument(
        '--episodes', type=int, default=1,
        help='Number of episodes during testing.')
    args = parser.parse_args()

    # Set the experiment parameter
    PARAMS_FILE = str(Path(os.path.join(PARAMS_DIR, args.params)))
    params = load_params(PARAMS_FILE)
    params['mode'] = args.mode
    params['test_episodes'] = args.episodes
    params['random_seed'] = seed_generator(params['random_seed'], params['runs'])
    params['start_time'] = start_time

    # Set up directory structure if training
    #if params['mode'] == 'train':
    # Create experiment dir
    params['exp_dir'] = create_dir(
        Path(os.path.join(
            LOG_DIR, 
            params['env_type'], 
            params['env_name'],
            str(time.strftime("%Y-%m-%d_%H-%M")))))
    # Safe experiment parameters to log dir
    copy_file(
        PARAMS_FILE,
        str(Path(os.path.join(params['exp_dir'], args.params))))
    # Add commit version for reproducability
    label = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    f = open(os.path.join(params['exp_dir'], 'RL2go_commit-version.txt'), 'w+')
    f.write('{}\r\n'.format(label))
    f.close()
    
    # Prepare the tensorflow configuration settings
    # TODO: Make better suitable for multicore processing
    params['cores'] = 1
    #params['gpus'] = len(get_available_gpus()) if params['use_gpu'] else 0
    params['gpus'] = 1 if params['use_gpu'] else 0
    params['tf_config'] = tf.ConfigProto(
        intra_op_parallelism_threads=params['cores'],
        inter_op_parallelism_threads=params['cores'],
        allow_soft_placement=True,
        device_count={'CPU':params['cores'], 'GPU':params['gpus']})
    if params['use_gpu']:
        params['tf_config'].gpu_options.allow_growth = True
        params['tf_config'].gpu_options.visible_device_list = params['gpu_device']

    # Run the experiment
    main(params)
