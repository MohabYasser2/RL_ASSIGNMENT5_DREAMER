"""
Environment helpers and wrappers.

This module provides convenience functions used in `main.py` to adapt Gym
observations for convolutional encoders and to normalise the newer
`gymnasium` API (terminated/truncated -> done).
"""

import gymnasium as gym
import numpy as np


def getEnvProperties(env):
    """Return canonical environment properties used across the codebase.

    Returns: (observationShape, isDiscrete, actionSize, actionLow, actionHigh)
    """
    observationShape = env.observation_space.shape
    if isinstance(env.action_space, gym.spaces.Discrete):
        isDiscrete = True
        actionSize = env.action_space.n
        actionLow = None
        actionHigh = None
    elif isinstance(env.action_space, gym.spaces.Box):
        isDiscrete = False
        actionSize = env.action_space.shape[0]
        actionLow = env.action_space.low.tolist()
        actionHigh = env.action_space.high.tolist()
    else:
        raise ValueError(f"Unsupported action space type: {type(env.action_space)}")
    return observationShape, isDiscrete, actionSize, actionLow, actionHigh


class GymPixelsProcessingWrapper(gym.ObservationWrapper):
    """Transpose HWC images to CHW and normalize to [0,1].

    Many conv nets expect (C, H, W). This wrapper converts observations and
    scales pixel values from [0,255] -> [0,1].
    """
    def __init__(self, env):
        super().__init__(env)
        observationSpace = self.observation_space
        newObsShape = observationSpace.shape[-1:] + observationSpace.shape[:2]
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=newObsShape, dtype=np.float32)

    def observation(self, observation):
        observation = np.transpose(observation, (2, 0, 1))/255.0
        return observation
    

class CleanGymWrapper(gym.Wrapper):
    """Compatibility wrapper for `gymnasium` step/reset signatures.

    Converts `(obs, reward, terminated, truncated, info)` -> `(obs, reward, done)`.
    Reset returns just the observation to keep the rest of the code simple.
    """
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return obs, reward, done

    def reset(self, seed=None):
        obs, info = self.env.reset(seed=seed)
        return obs