import torch
import numpy as np
import random
import os
import gym
from typing import Optional, Tuple


class RNG:
    rng: np.random.Generator = None
    #RNG 类定义了一个类变量 rng，用于存储一个 NumPy 随机数生成器 


def set_global_seed(seed: int, *args: Tuple[gym.Env]) -> None:
    """Sets seed for PyTorch, NumPy, and random.

    Args:
        seed: The random seed to use.
        args: The gym environment(s) to seed.
    """
    random.seed(seed)
    tseed = random.randint(1, 1e6)
    npseed = random.randint(1, 1e6)
    ospyseed = random.randint(1, 1e6)
    torch.manual_seed(tseed)
    np.random.seed(npseed)
    for env in args:
        env.seed(seed=seed)
        env.observation_space.seed(seed=seed)
        env.action_space.seed(seed=seed)
    os.environ["PYTHONHASHSEED"] = str(ospyseed)
    RNG.rng = np.random.Generator(np.random.PCG64(seed=seed))
