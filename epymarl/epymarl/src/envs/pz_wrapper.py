from pathlib import Path
import importlib

import gymnasium as gym
from gymnasium.spaces import Tuple

import pettingzoo


class PettingZooWrapper(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 5,
    }

    def __init__(self, env_or_lib_name, env_name=None, **kwargs):
        """包装 PettingZoo 环境
        
        Args:
            env_or_lib_name: 可以是 PettingZoo 环境实例或环境库名称
            env_name: 环境名称（如果 env_or_lib_name 是库名称，则需要提供）
            **kwargs: 其他参数
        """
        if isinstance(env_or_lib_name, str):
            # 如果传入的是字符串，则从 pettingzoo 包导入环境
            if env_name is None:
                raise ValueError("当 env_or_lib_name 是字符串时，必须提供 env_name")
            env = importlib.import_module(f"pettingzoo.{env_or_lib_name}.{env_name}")
            self._env = env.parallel_env(**kwargs)
        else:
            # 如果传入的是环境实例，直接使用
            self._env = env_or_lib_name

        self._env.reset()

        self.n_agents = len(self._env.agents)
        self.last_obs = None

        self.action_space = Tuple(
            tuple([self._env.action_spaces[k] for k in self._env.agents])
        )
        self.observation_space = Tuple(
            tuple([self._env.observation_spaces[k] for k in self._env.agents])
        )

    def reset(self, *args, **kwargs):
        """重置环境"""
        obs = self._env.reset(*args, **kwargs)
        if isinstance(obs, tuple):
            obs, info = obs
        else:
            info = {}
        obs = tuple([obs[k] for k in self._env.agents])
        self.last_obs = obs
        return obs, info

    def render(self, mode="human"):
        return self._env.render(mode)

    def step(self, actions):
        dict_actions = {}
        for agent, action in zip(self._env.agents, actions):
            dict_actions[agent] = action

        observations, rewards, dones, truncated, infos = self._env.step(dict_actions)

        obs = tuple([observations[k] for k in self._env.agents])
        rewards = [rewards[k] for k in self._env.agents]
        done = all([dones[k] for k in self._env.agents])
        truncated = all([truncated[k] for k in self._env.agents])
        info = {
            f"{k}_{key}": value
            for k in self._env.agents
            for key, value in infos[k].items()
        }
        if done:
            # empty obs and rewards for PZ environments on terminated episode
            assert len(obs) == 0
            assert len(rewards) == 0
            obs = self.last_obs
            rewards = [0] * len(obs)
        else:
            self.last_obs = obs
        return obs, rewards, done, truncated, info

    def close(self):
        return self._env.close()

    def seed(self, seed=None):
        """设置随机种子"""
        if hasattr(self._env, 'seed'):
            return self._env.seed(seed)
        return None


# 注册所有 PettingZoo 环境
envs = Path(pettingzoo.__path__[0]).glob("**/*_v?.py")
for e in envs:
    name = e.stem.replace("_", "-")
    lib = e.parent.stem
    filename = e.stem

    gymkey = f"pz-{lib}-{name}"
    gym.register(
        gymkey,
        entry_point="envs.pz_wrapper:PettingZooWrapper",
        kwargs={
            "env_or_lib_name": lib,
            "env_name": filename,
        },
    )
