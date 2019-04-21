# coding=utf-8
# Copyright 2018 The Dopamine Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A lightweight logging mechanism for dopamine agents."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
import tensorflow as tf


#CHECKPOINT_DURATION = 4


class Logger(object):
    """Class for maintaining a dictionary of data to log."""

    def __init__(self, logging_dir, logs_to_keep=4):
        """Initializes Logger.

        Parameters
        ----------
            logging_dir : str
                Directory to which logs are written.
        """
        # Dict used by logger to store data.
        self.data = {}
        self._logging_enabled = True
        self._logs_to_keep = logs_to_keep

        if not logging_dir:
            tf.logging.info('Logging directory not specified, will not log.')
            self._logging_enabled = False
            return
        # Try to create logging directory.
        try:
            tf.gfile.MakeDirs(logging_dir)
        except tf.errors.PermissionDeniedError:
            # If it already exists, ignore exception.
            pass
        if not tf.gfile.Exists(logging_dir):
            tf.logging.warning(
                'Could not create directory %s, logging will be disabled.',
                logging_dir)
            self._logging_enabled = False
            return
        self._logging_dir = logging_dir

    def __setitem__(self, key, value):
        """This method will set an entry at key with value in the dictionary.
        It will effectively overwrite any previous data at the same key.
        
        Parameters
        ----------
            key : str
                Indicating key where to write the entry.
            value : Object
                Python object to store.
        """
        if self._logging_enabled:
            self.data[key] = value

    def _generate_filename(self, filename_prefix, era_number):
        filename = '{}_{}'.format(filename_prefix, era_number)
        return os.path.join(self._logging_dir, filename)

    def log_to_file(self, filename_prefix, era_number):
        """Save the pickled dictionary to a file.
        
        Parameters
        ----------
            filename_prefix : str
                Name of the file to use (without era number).
            era_number : int
                era number, appended to the end of filename_prefix.
        """
        if not self._logging_enabled:
            tf.logging.warning('Logging is disabled.')
            return
        log_file = self._generate_filename(filename_prefix, era_number)
        with tf.gfile.GFile(log_file, 'w') as fout:
            pickle.dump(self.data, fout, protocol=pickle.HIGHEST_PROTOCOL)
        # After writing a checkpoint file, we garbage collect the log file
        # that is logs_to_keep versions old.
        stale_era_number = era_number - self._logs_to_keep
        if stale_era_number >= 0:
            stale_file = self._generate_filename(filename_prefix,
                                            stale_era_number)
            try:
                tf.gfile.Remove(stale_file)
            except tf.errors.NotFoundError:
                # Ignore if file not found.
                pass

    def is_logging_enabled(self):
        """Return if logging is enabled."""
        return self._logging_enabled


class Statistics(object):
    """A class for storing metrics.
    The internal format is as follows: we maintain a mapping from keys to lists.
    Each list contains all the values corresponding to the given key.
    For example, self.data_lists['train_episode_returns'] might contain the
        per-episode returns achieved during this iteration.
    Attributes:
        data_lists: dict mapping each metric_name (str) to a list of said metric
        across episodes.
    """

    def __init__(self):
        self.data_lists = {}

    def append(self, data_pairs):
        """Add the given values to their corresponding key-indexed lists.
        Args:
        data_pairs: A dictionary of key-value pairs to be recorded.
        """
        for key, value in data_pairs.items():
            if key not in self.data_lists:
                self.data_lists[key] = []
            self.data_lists[key].append(value)