from collections.abc import Iterable
import warnings

import gymnasium as gym
from gymnasium.spaces import flatdim
from gymnasium.wrappers import TimeLimit
import numpy as np

from .multiagentenv import MultiAgentEnv
from .wrappers import FlattenObservation
import envs.pretrained as pretrained  # noqa

try:
    from .pz_wrapper import PettingZooWrapper  # noqa
except ImportError:
    warnings.warn(
        "PettingZoo is not installed, so these environments will not be available! To install, run `pip install pettingzoo`"
    )

try:
    from .vmas_wrapper import VMASWrapper  # noqa
except ImportError:
    warnings.warn(
        "VMAS is not installed, so these environments will not be available! To install, run `pip install 'vmas[gymnasium]'`"
    )

try:
    from .sumo_wrapper import parallel_env   # noqa
except ImportError:
    warnings.warn(
        "somthing went wrong with sumo_wrapper import, please check the code"
    )

try:
    from .ev_wrapper import parallel_env   # noqa
except ImportError:
    warnings.warn(
        "somthing went wrong with ev_wrapper import, please check the code"
    )
try:
    from .test_wrapper import EVPZWrapper   # noqa
except ImportError:
    warnings.warn(
        "somthing went wrong with ev_wrapper import, please check the code"
    )


class GymmaWrapper(MultiAgentEnv):
    def __init__(
        self,
        key,
        time_limit,
        pretrained_wrapper,
        seed,
        common_reward,
        reward_scalarisation,
        **kwargs,
    ):
        self._env = gym.make(f"{key}", **kwargs)
        self._env = TimeLimit(self._env, max_episode_steps=time_limit)
        self._env = FlattenObservation(self._env)

        if pretrained_wrapper:
            self._env = getattr(pretrained, pretrained_wrapper)(self._env)

        # 获取代理数量
        # if hasattr(self._env, 'agents'):
        #     self.n_agents = len(self._env.agents)
        # else:
        self.n_agents = self._env.unwrapped.n_agents

        self.episode_limit = time_limit
        self._obs = None
        self._info = None

        # # 获取动作空间和观察空间
        # if hasattr(self._env, 'action_spaces'):
        #     self.longest_action_space = max(self._env.action_space)
        #     self.longest_observation_space = max(self._env.observation_space)
        # else:
        self.longest_action_space = max(self._env.action_space, key=lambda x: x.n)
        self.longest_observation_space = max(self._env.observation_space, key=lambda x: x.shape)

        self._seed = seed
        try:
            self._env.unwrapped.seed(self._seed)
        except:
            self._env.reset(seed=self._seed)

        self.common_reward = common_reward
        if self.common_reward:
            if reward_scalarisation == "sum":
                self.reward_agg_fn = lambda rewards: sum(rewards)
            elif reward_scalarisation == "mean":
                self.reward_agg_fn = lambda rewards: sum(rewards) / len(rewards)
            else:
                raise ValueError(
                    f"Invalid reward_scalarisation: {reward_scalarisation} (only support 'sum' or 'mean')"
                )

    def _pad_observation(self, obs):
        return [
            np.pad(
                o,
                (0, self.longest_observation_space.shape[0] - len(o)),
                "constant",
                constant_values=0,
            )
            for o in obs
        ]

    def step(self, actions):
        """Returns obss, reward, terminated, truncated, info"""
        actions = [int(a) for a in actions]
        # if hasattr(self._env, 'agents'):
        #     action_dict = {agent: action for agent, action in zip(self._env.agents, actions)}
        #     obs, reward, done, truncated, self._info = self._env.step(action_dict)
        #     obs = [obs[agent] for agent in self._env.agents]
        #     reward = [reward[agent] for agent in self._env.agents]
        #     done = [done[agent] for agent in self._env.agents]
        #     truncated = [truncated[agent] for agent in self._env.agents]
        # else:
        obs, reward, done, truncated, self._info = self._env.step(actions)
            
        self._obs = self._pad_observation(obs)

        # if self.common_reward and isinstance(reward, Iterable):
        #     reward = float(self.reward_agg_fn(reward))
        # elif not self.common_reward and not isinstance(reward, Iterable):
        #     warnings.warn(
        #         "common_reward is False but received scalar reward from the environment, returning reward as is"
        #     )

        if isinstance(done, Iterable):
            done = all(done)
        return self._obs, reward, done, truncated, self._info

    def get_obs(self):
        """Returns all agent observations in a list"""
        return self._obs

    def get_obs_agent(self, agent_id):
        """Returns observation for agent_id"""
        raise self._obs[agent_id]

    def get_obs_size(self):
        """Returns the shape of the observation"""
        return flatdim(self.longest_observation_space)

    def get_state(self):
        return np.concatenate(self._obs, axis=0).astype(np.float32)

    def get_state_size(self):
        """Returns the shape of the state"""
        if hasattr(self._env.unwrapped, "state_size"):
            return self._env.unwrapped.state_size
        return self.n_agents * flatdim(self.longest_observation_space)

    def get_avail_actions(self):
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)
        return avail_actions

    # def get_avail_agent_actions(self, agent_id):
    #     """Returns the available actions for agent_id"""
    #     # if hasattr(self._env, 'agents'):
    #     #     agent = self._env.agents[agent_id]
    #     #     valid = flatdim(self._env.action_spaces[agent]) * [1]
    #     # else:
    #     valid = flatdim(self._env.action_space[agent_id]) * [1]
    #     invalid = [0] * (self.longest_action_space.n - len(valid))
    #     return valid + invalid
    def get_avail_agent_actions(self, agent_id):
        """Returns the available actions for agent_id"""
        if hasattr(self._env, 'get_avail_agent_actions'):
            # 获取可用动作索引列表
            avail_actions = self._env.get_avail_agent_actions(agent_id)
            # 创建掩码
            mask = [0] * self.get_total_actions()
            # 将可用动作的索引位置设为1
            for idx in avail_actions:
                if idx < len(mask):
                    mask[idx] = 1
            return mask
        else:
            # 默认情况：所有动作都可用
            return [1] * self.get_total_actions()

    def get_total_actions(self):
        """Returns the total number of actions an agent could ever take"""
        # TODO: This is only suitable for a discrete 1 dimensional action space for each agent
        return flatdim(self.longest_action_space)

    def reset(self, seed=None, options=None):
        """Returns initial observations and info"""
        obs, info = self._env.reset(seed=seed, options=options)
        self._obs = self._pad_observation(obs)
        return self._obs, info

    def render(self):
        self._env.render()

    def close(self):
        self._env.close()

    def seed(self, seed=None):
        return self._env.unwrapped.seed(seed)

    def save_replay(self):
        pass

    def get_stats(self):
        return {}
