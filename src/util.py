import os
import shutil
import sys
from distutils.dir_util import copy_tree
from pathlib import Path

import yaml
import numpy as np
from tensorflow.python.client import device_lib

def create_dir(path_to_dir):
    """Create a directory at the given path if possible.

    Parameters
    ----------
    path_to_dir : Path
        Path to the directory that should be created if it does not exist.

    Returns
    -------
    path_to_dir : Path
        Verified path to existing directory.
    """
    if not path_to_dir.is_dir():
        try:
            path_to_dir.mkdir(mode=0o755, parents=True, exist_ok=False)
        except:
            print('Error when creating directory: {}'.format(path_to_dir))
            sys.exit()
    return str(path_to_dir)

def copy_dir(source_dir, target_dir):
    """Create a directory at the given path if possible.

    Parameters
    ----------
    source_dir : Path
        Path to the directory that should be created if it does not exist.

    Returns
    -------
    path_to_dir : Path
        Verified path to existing directory.
    """
    try:
        copy_tree(str(source_dir), str(target_dir))
    except:
        print('Error when copying directory: {} -> {}'.format(
            source_dir, target_dir))
        sys.exit()

def copy_file(src, dest):
    """Copies a file from a source to a destination directory.

    Parameters
    ----------
    src : Path
        Path to the source file that needs to be copied.

    dest : Path
        Path to the destination file where we want a copy of the source.

    Returns
    -------
    True : bool
        Returns true if successful copied the file.
    """
    try:
        shutil.copy(src, dest)
    # eg. src and dest are the same file
    except shutil.Error as e:
        #_logger.critical("Can't copy file - %s" % str(e))
        sys.exit(e)
    # eg. source or destination doesn't exist
    except IOError as e:
        #_logger.critical("Can't copy file - %s" % str(e))
        sys.exit(e) 

def load_params(yaml_file):
    """Loads the experiment parameter from file. 

    Parameters
    ----------
    yaml_file : Path
        Location of the parameter file in yaml format.

    Returns
    -------
    params : dict
        Dictionary containing all experiment parameters.
    """
    with open(yaml_file, 'r') as yamlfile:
        params = yaml.load(yamlfile, Loader=yaml.Loader)
    return params

def seed_generator(seed, runs):
    """Generates a sequenze of numbers starting with the original seed to have
    a consistent seed setting for different runs.

    Parameters
    ----------
    seed : int
        Original seed set by the user (also used for first run).
    runs : int
        Number of seeds that are needed for all runs in the experiment.

    Returns
    -------
    seed_list : list of int
        Generated list of seeds, one for each run in the experiment.
    """
    original = seed
    seed_list = [seed]
    for _ in range(1, runs):
        if seed > 100000:
            while original in seed_list:
                original += 1
            seed = original
        else:
            seed = int(seed * 7/5)
        seed_list.append(seed)
    return seed_list

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

def get_object_config(o):
    """Extracts configuration information from an object.

    Parameters
    ----------
    o : Object
        Initialized object from which we have no configuration.

    Returns
    -------
    config: dict
        Dictionary containing the configuration information of
        the object.
    """
    if o is None:
        return None
    config = {
        'class_name': o.__class__.__name__,
        'config': o.get_config()
    }
    return config

def get_readable_time(seconds):
    """ Returns a human readable time string from seconds. 
    
    Parameters
    ----------
        seconds : int
            Time in seconds.

    Returns
    -------
        text : str
            Time as human readable string.
    """
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 23:
        d, h = divmod(h, 24)
        return '{:02d}d:{:02d}:{:02d}:{:02d}'.format(d, h, m, s)
    return '{:02d}:{:02d}:{:02d}'.format(h, m, s)
  
def estimate_remaining_time(seconds, progress):
    """ Estimates remaining time of a process. 
    
    Parameters
    ----------
        seconds : int
            Time in seconds.
        progress : float
            Progress of the process.

    Returns
    -------
        remaining : int
            Estimated remaining time in seconds.
    """
    delta = 0.0001
    return int(int(seconds) * (1 - progress) / (progress + delta))

def summarize_data(data, summary_keys, prefix='era'):
    """Processes log data into a per-iteration summary.
    
    Parameters
    ----------
        data : dict
            Dictionary loaded by load_statistics describing the data. This
            dictionary has keys era_0, era_1, ... describing per-era data.
        summary_keys : list of keys
            List of per-era data keys to be summarized from data dict.
    
    Returns
    -------
        summary : dict
            Dictionary mapping each key in summary_keys to a per-era summary.

    Example
    -------
        data = load_statistics(...)
        summarize_data(
            data, ['trainEpisodeSteps',
            'evalEpisodeSteps'])    
    """
    summary = {}
    latest_iteration_number = len(data.keys())
    current_value = None

    for key in summary_keys:
        summary[key] = []
        # Compute per-iteration average of the given key.
        for i in range(latest_iteration_number):
            iter_key = '{}{}'.format(prefix, i)
            # We allow reporting the same value multiple times when data is missing.
            # If there is no data for this iteration, use the previous'.
            if iter_key in data:
                current_value = np.mean(data[iter_key][key])
            summary[key].append(current_value)
    return summary