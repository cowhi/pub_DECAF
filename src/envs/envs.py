
import os
import random
import sys
from collections import deque

import gym
from gym import spaces
import cv2
cv2.ocl.setUseOpenCL(False)  # pylint: disable=no-member
import numpy as np


def prepare_env(params, seed):
    """Prepares an environment according to the settings in the
    parameter dictionary.

    Parameters
    ----------
    params : dict
        Directory that contains all environment settings.
    seed : int
        Random seed for this instance of the environment.

    Returns
    -------
    env : Env
        Environment as initialized with the parameter specifications.
    """
    env_name = params['env_name']
    if params['env_type'] == 'Atari':
        game_version = 'v0' if params['sticky_actions'] else 'v4'
        env_name = '{}NoFrameskip-{}'.format(params['env_name'], game_version)
    env = gym.make(env_name)
    env.seed(seed)
    if params['env_type'] == 'Atari':
        if params['blur']:
            return AtariBlurPreprocessing(
                env.env,
                blur_area=params['blur_area'],
                blur_color=params['blur_color'],
                blur_color_render=params['blur_color_render'])
        return AtariPreprocessing(env.env)

    return env_wrapper(
        env=env,
        warp_size=params['warp_size'], 
        frame_stack=params['frame_stack'],
        episode_life=params['episode_life'],
        max_episode_steps=params['max_steps_episode'],
        init_noops=params['init_noops'],
        clip_reward=params['clip_reward'],
        skip_frames=params['skip_frames'],
        seed=seed)

def env_wrapper(env, warp_size=None, frame_stack=None, episode_life=False,
        max_episode_steps=None, init_noops=None, clip_reward=None, skip_frames=None, seed=123):
    """Configure environment for DeepMind-style Atari.
    """
    if episode_life:
        env = EpisodicLifeEnv(env)
    if init_noops is not None:
        env = NoopResetEnv(env, noop_max=init_noops, seed=seed)
    if skip_frames is not None:
        env = MaxAndSkipEnv(env, skip=skip_frames)
    if warp_size is not None:
        env = WarpFrame(env, warp=warp_size)
    if max_episode_steps is not None:
        env = StepLimitEnv(env, max_episode_steps=max_episode_steps)
    try:
        if 'FIRE' in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
    except Exception: 
        pass
    if frame_stack is not None:
        env = FrameStack(env, stack=frame_stack)
    #env = NumpyEnv(env)
    if clip_reward is not None:
        env = ClipRewardEnv(env, border=clip_reward)
    return env


class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30, seed=123):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        gym.Wrapper.__init__(self, env)
        self.seed = seed
        np.random.seed(self.seed)
        random.seed(self.seed)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'
        print('NoopResetEnv (Initializes episodes with max {} NOOPs)'.format(noop_max))

    def reset(self, **kwargs):  # pylint: disable=E0202
        """ Do no-op action for a number of steps in [1, noop_max]."""
        #print('RESET NoopResetEnv START')
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.randint(1, self.noop_max + 1)  # pylint: disable=E1101
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        #print('RESET NoopResetEnv END (noops: {}) {}'.format(noops, obs.shape))
        return obs

    def step(self, action):  # pylint: disable=E0202
        #print('STEP NoopResetEnv')
        return self.env.step(action)


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,)+env.observation_space.shape, dtype=np.uint8)
        self._skip       = skip
        print('MaxAndSkipEnv (Only keep 1 frame out of {})'.format(skip))

    def step(self, action):  # pylint: disable=E0202
        """Repeat action, sum reward, and max over last observations."""
        #print('STEP MaxAndSkipEnv START')
        total_reward = 0.0
        accumulated_info = {}
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2: self._obs_buffer[0] = obs
            if i == self._skip - 1: self._obs_buffer[1] = obs
            total_reward += reward
            for key, value in info.items():
                if not np.isreal(value):
                    continue
                if key not in accumulated_info:
                    accumulated_info[key] = np.zeros_like(value)
                accumulated_info[key] += value
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)
        #print('STEP MaxAndSkipEnv END:', max_frame.shape)
        return max_frame, total_reward, done, accumulated_info

    def reset(self, **kwargs):  # pylint: disable=E0202
        #print('RESET MaxAndSkipEnv')
        return self.env.reset(**kwargs)


class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done  = True
        print('EpisodicLifeEnv (Reset only when all lifes are exhausted)')

    def step(self, action):  # pylint: disable=E0202
        #print('STEP EpisodicLifeEnv START')
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condition for a few frames
            # so it's important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        #print('STEP EpisodicLifeEnv END:', obs.shape)
        return obs, reward, done, info

    def reset(self, **kwargs):  # pylint: disable=E0202
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        #print('RESET EpisodicLifeEnv START')
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        #print('RESET EpisodicLifeEnv END:', obs.shape)
        return obs


class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        """Take action on reset for environments that are fixed until firing."""
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3
        print('FireResetEnv (Starts episode with FIRE button)')

    def reset(self, **kwargs):  # pylint: disable=E0202
        #print('RESET FireResetEnv START')
        self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset(**kwargs)
        #print('RESET FireResetEnv END:', obs.shape)
        return obs

    def step(self, action):  # pylint: disable=E0202
        #print('STEP FireResetEnv')
        return self.env.step(action)


class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env, warp=84, grayscale=True):
        """Warp frames to 84x84 as done in the Nature paper and later work."""
        gym.ObservationWrapper.__init__(self, env)
        self.width = warp
        self.height = warp
        self.grayscale = grayscale
        old_shape = env.observation_space.shape
        if self.grayscale:
            self.observation_space = spaces.Box(low=0, high=255,
                shape=(self.height, self.width, 1), dtype=np.uint8)
        else:
            self.observation_space = spaces.Box(low=0, high=255,
                shape=(self.height, self.width, 3), dtype=np.uint8)
        new_shape = self.observation_space.shape
        print("WarpFrame", old_shape, '-->', new_shape)

    def observation(self, frame):
        #print('OBSERVATION WarpFrame START:', frame.shape)
        if self.grayscale:
            frame = cv2.cvtColor(  # pylint: disable=no-member
                frame, 
                cv2.COLOR_RGB2GRAY)  # pylint: disable=no-member
        frame = cv2.resize(  # pylint: disable=no-member
            frame, 
            (self.width, self.height), 
            interpolation=cv2.INTER_AREA)  # pylint: disable=no-member
        if self.grayscale:
            frame = np.expand_dims(frame, -1)
        #print('OBSERVATION WarpFrame END:', frame.shape)
        return frame


class ClipRewardEnv(gym.RewardWrapper):
    def __init__(self, env, border=1):
        gym.RewardWrapper.__init__(self, env)
        self.border = border
        print('ClipRewardEnv ({} < reward < {})'.format(-self.border, self.border))

    def reward(self, reward):
        """Bin reward to [-1, 1] by its sign."""
        #"""Bin reward to {+1, 0, -1} by its sign."""
        #print('REWARD ClipRewardEnv ({} < {} < {})'.format(-self.border, reward, self.border))
        return reward if -self.border < reward < self.border else np.sign(reward) * self.border


class FrameStack(gym.Wrapper):
    def __init__(self, env, stack=4):
        """Stack k last frames.
        Returns lazy array, which is much more memory efficient.
        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = stack
        self.frames = deque([], maxlen=self.k)
        #old_shape = env.observation_space.shape
        #new_shape = ((old_shape[-1],) + old_shape[:-1] )
        #new_shape = (old_shape[:-1] + (old_shape[-1] * self.k,))
        #self.observation_space = spaces.Box(low=0, high=255, shape=new_shape, dtype=env.observation_space.dtype)
        print("FrameStack", self.k, '*', env.observation_space.shape)

    def reset(self):  # pylint: disable=E0202
        #print('RESET FrameStack START')
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        #print('RESET FrameStack END:', len(self.frames), ob.shape)
        return self._get_ob()

    def step(self, action):  # pylint: disable=E0202
        #print('STEP FrameStack START')
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        #print('STEP FrameStack END:', len(self.frames), ob.shape)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))


class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.
        This object should only be converted to numpy array before being passed to the model.
        """
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=-1)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]

class NumpyEnv(gym.ObservationWrapper):
    """
    Change image shape to CWH
    """
    def __init__(self, env, frames=4):
        super(NumpyEnv, self).__init__(env)
        old_shape = self.observation_space.shape
        if len(old_shape) == 3:
            #self.output_shape = ((old_shape[-1],) + old_shape[:-1] )
            self.observation_space.shape = ((frames,) + old_shape[:-1] )
        elif len(old_shape) == 1:
            #self.output_shape = (1, old_shape[0])
            self.observation_space.shape = (1, old_shape[0])
        else:
            print("Can't handle this type of observation space:", old_shape, len(old_shape))
            sys.exit()
        print("NumpyEnv", old_shape, '-->', self.observation_space.shape)
        
    def observation(self, observation):
        #print('OBSERVATION NumpyEnv START', len(observation), observation[0].shape)
        if not isinstance(observation, np.ndarray):
            # This works for stacked 2D (grayscale) images 
            result = np.swapaxes(observation, 2, 0)
            #print('OBSERVATION 1NumpyEnv END', result.shape)
            return result
        # This works for an array of values (as in classic domain for example)
        result = np.expand_dims(observation, axis=0)
        #print('OBSERVATION 2NumpyEnv END', result.shape)
        return result


class StepLimitEnv(gym.Wrapper):
    def __init__(self, env, max_episode_steps=None):
        super(StepLimitEnv, self).__init__(env)
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = 0
        print("StepLimitEnv", max_episode_steps)

    def _past_limit(self):
        """Return true if we are past our limit"""
        if self._max_episode_steps is not None and self._max_episode_steps <= self._elapsed_steps:
            gym.logger.debug("Env has passed the step limit defined by StepLimit.")
            return True
        return False

    def step(self, action): # pylint: disable=E0202
        #gym.logger.debug("RESET StepLimitEnv START")
        observation, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1
        if self._past_limit():
            done = True 
        #gym.logger.debug("RESET StepLimitEnv END", observation.shape)
        return observation, reward, done, info

    def reset(self):  # pylint: disable=E0202
        #gym.logger.debug("RESET StepLimitEnv START")
        self._elapsed_steps = 0
        #gym.logger.debug("RESET StepLimitEnv END")
        return self.env.reset()



class AtariPreprocessing(object):
    """A class implementing image preprocessing for Atari 2600 agents.
    Specifically, this provides the following subset from the JAIR paper
    (Bellemare et al., 2013) and Nature DQN paper (Mnih et al., 2015):
        * Frame skipping (defaults to 4).
        * Terminal signal when a life is lost (off by default).
        * Grayscale and max-pooling of the last two frames.
        * Downsample the screen to a square image (defaults to 84x84).
    More generally, this class follows the preprocessing guidelines set down in
    Machado et al. (2018), "Revisiting the Arcade Learning Environment:
    Evaluation Protocols and Open Problems for General Agents".
    """

    def __init__(self, environment, frame_skip=4, terminal_on_life_loss=False,
               screen_size=84):
        """Constructor for an Atari 2600 preprocessor.
        Args:
            environment: Gym environment whose observations are preprocessed.
            frame_skip: int, the frequency at which the agent experiences the game.
            terminal_on_life_loss: bool, If True, the step() method returns
            is_terminal=True whenever a life is lost. See Mnih et al. 2015.
            screen_size: int, size of a resized Atari 2600 frame.
        Raises:
            ValueError: if frame_skip or screen_size are not strictly positive.
        """
        if frame_skip <= 0:
            raise ValueError(
                'Frame skip should be strictly positive, got {}'.format(
                    frame_skip))
        if screen_size <= 0:
            raise ValueError(
                'Target screen size should be strictly positive, got {}'.format(
                    screen_size))

        self.environment = environment
        self.terminal_on_life_loss = terminal_on_life_loss
        self.frame_skip = frame_skip
        self.screen_size = screen_size

        obs_dims = self.environment.observation_space
        # Stores temporary observations used for pooling over two successive
        # frames.
        self.screen_buffer = [
            np.empty((obs_dims.shape[0], obs_dims.shape[1]), dtype=np.uint8),
            np.empty((obs_dims.shape[0], obs_dims.shape[1]), dtype=np.uint8)
        ]

        self.game_over = False
        self.lives = 0  # Will need to be set by reset().

    @property
    def observation_space(self):
        # Return the observation space adjusted to match the shape of the processed
        # observations.
        return spaces.Box(
            low=0, 
            high=255, 
            shape=(self.screen_size, self.screen_size, 1),
            dtype=np.uint8)

    @property
    def action_space(self):
        return self.environment.action_space

    @property
    def reward_range(self):
        return self.environment.reward_range

    @property
    def metadata(self):
        return self.environment.metadata

    def reset(self):
        """Resets the environment.
        Returns:
        observation: numpy array, the initial observation emitted by the
            environment.
        """
        self.environment.reset()
        self.lives = self.environment.ale.lives()
        self._fetch_grayscale_observation(self.screen_buffer[0])
        self.screen_buffer[1].fill(0)
        return self._pool_and_resize()

    def render(self, mode):
        """Renders the current screen, before preprocessing.
        This calls the Gym API's render() method.
        Args:
            mode: Mode argument for the environment's render() method.
                Valid values (str) are:
                'rgb_array': returns the raw ALE image.
                'human': renders to display via the Gym renderer.
        Returns:
            if mode='rgb_array': numpy array, the most recent screen.
            if mode='human': bool, whether the rendering was successful.
        """
        return self.environment.render(mode)

    def step(self, action):
        """Applies the given action in the environment.
        Remarks:
            * If a terminal state (from life loss or episode end) is reached, this may
            execute fewer than self.frame_skip steps in the environment.
        *    Furthermore, in this case the returned observation may not contain valid
            image data and should be ignored.
        Args:
            action: The action to be executed.
        Returns:
            observation: numpy array, the observation following the action.
            reward: float, the reward following the action.
            is_terminal: bool, whether the environment has reached a terminal state.
                This is true when a life is lost and terminal_on_life_loss, or when the
                episode is over.
            info: Gym API's info data structure.
        """
        accumulated_reward = 0.

        for time_step in range(self.frame_skip):
            # We bypass the Gym observation altogether and directly fetch the
            # grayscale image from the ALE. This is a little faster.
            _, reward, game_over, info = self.environment.step(action)
            accumulated_reward += reward

            if self.terminal_on_life_loss:
                new_lives = self.environment.ale.lives()
                is_terminal = game_over or new_lives < self.lives
                self.lives = new_lives
            else:
                is_terminal = game_over

            if is_terminal:
                break
            # We max-pool over the last two frames, in grayscale.
            elif time_step >= self.frame_skip - 2:
                t = time_step - (self.frame_skip - 2)
                self._fetch_grayscale_observation(self.screen_buffer[t])

        # Pool the last two observations.
        observation = self._pool_and_resize()

        self.game_over = game_over
        return observation, accumulated_reward, is_terminal, info

    def _fetch_grayscale_observation(self, output):
        """Returns the current observation in grayscale.
        The returned observation is stored in 'output'.
        Args:
        output: numpy array, screen buffer to hold the returned observation.
        Returns:
        observation: numpy array, the current observation in grayscale.
        """
        self.environment.ale.getScreenGrayscale(output)
        return output

    def _pool_and_resize(self):
        """Transforms two frames into a Nature DQN observation.
        For efficiency, the transformation is done in-place in self.screen_buffer.
        Returns:
        transformed_screen: numpy array, pooled, resized screen.
        """
        # Pool if there are enough screens to do so.
        if self.frame_skip > 1:
            np.maximum(
                self.screen_buffer[0], 
                self.screen_buffer[1],
                out=self.screen_buffer[0])

        transformed_image = cv2.resize(  # pylint: disable=no-member
            self.screen_buffer[0],
            (self.screen_size, self.screen_size),
            interpolation=cv2.INTER_AREA)  # pylint: disable=no-member
        int_image = np.asarray(transformed_image, dtype=np.uint8)
        return np.expand_dims(int_image, axis=2)


class AtariBlurPreprocessing(AtariPreprocessing):
    """A class implementing image preprocessing for Pong. """

    def __init__(self, environment, frame_skip=4, terminal_on_life_loss=False,
               screen_size=84, blur_area=[0, 0, 0, 0], blur_color=0, blur_color_render=[0, 0, 0]):
        super(AtariBlurPreprocessing, self).__init__(
            environment=environment,
            frame_skip=frame_skip, 
            terminal_on_life_loss=terminal_on_life_loss,
            screen_size=screen_size)
        self.blur_area = blur_area
        self.blur_color = blur_color
        self.blur_color_render = blur_color_render
        self.viewer = None
        print('STarting blur agent')
    
    def _pool_and_resize(self):
        """Transforms two frames into a Nature DQN observation and blurs a given
        area within the image.
        For efficiency, the transformation is done in-place in self.screen_buffer.
        
        Returns
        -------
            transformed_screen : numpy array
                Transformed observation with blured area.
        """
        # Pool if there are enough screens to do so.
        if self.frame_skip > 1:
            np.maximum(
                self.screen_buffer[0], 
                self.screen_buffer[1],
                out=self.screen_buffer[0])

        transformed_image = cv2.resize(  # pylint: disable=no-member
            self.screen_buffer[0],
            (self.screen_size, self.screen_size),
            interpolation=cv2.INTER_AREA)  # pylint: disable=no-member
        
        transformed_image[
            self.blur_area[0]:self.blur_area[1],
            self.blur_area[2]:self.blur_area[3]] = self.blur_color
        
        int_image = np.asarray(transformed_image, dtype=np.uint8)
        return np.expand_dims(int_image, axis=2)

    def render(self, mode='human'):
        #img = self._get_image()
        img = self.environment.ale.getScreenRGB2()
        img = cv2.resize(  # pylint: disable=no-member
            img,
            (self.screen_size, self.screen_size),
            interpolation=cv2.INTER_AREA)
        img[
            self.blur_area[0]:self.blur_area[1],
            self.blur_area[2]:self.blur_area[3]] = self.blur_color_render
        img = cv2.resize(  # pylint: disable=no-member
            img,
            (630, 480),
            interpolation=cv2.INTER_AREA)
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
            return self.viewer.isopen