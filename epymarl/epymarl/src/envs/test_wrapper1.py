from pathlib import Path
import gymnasium as gym
from gymnasium.spaces import Tuple , Box , Discrete
import numpy as np
from .ev_envfmp import EVParallelEnv
class EVPZWrapper(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 5,
    }
    #传入EVPZWrapper类的obs 和 action space必须满足gym的要求
    #EVPZWrapper类包装后的obs 和 action space必须满足gymma的要求

    def __init__(self, **kwargs):
        self._env = EVParallelEnv(**kwargs)
        self.episode_limit = self._env.episode_limit
        self.n_agents = self._env.n_agents
        self.last_obs = None

        self.action_space = Tuple(
            tuple([self._env.action_spaces[k] for k in self._env.agents])
        )
        self.observation_space = Tuple(
            tuple([self._env.observation_spaces[k] for k in self._env.agents])
        )
        


    def reset(self, seed=None, options=None):
        """重置环境"""
        obs = self._env.reset(seed = seed)
        if isinstance(obs, tuple):
            obs, info = obs
        else:
            info = {}
        # obs = tuple([obs[k] for k in self._env.agents])
        obs = tuple([np.array(obs[k], dtype=np.float32) for k in self._env.agents])
        self.last_obs = obs
        return obs, info

    def render(self):
        return self._env.render()

    def step(self, actions):
        dict_actions = {}
        for agent, action in zip(self._env.agents, actions):
            agent_id = str(agent)
            dict_actions[agent_id] = action
            

        observations, rewards, dones, truncated, infos = self._env.step(dict_actions)

        obs = tuple([observations[k] for k in self._env.agents])
        rewards = [rewards[k] for k in self._env.agents]
        done = all([dones[k] for k in self._env.agents])
        truncated = all([truncated[k] for k in self._env.agents])
        # info = {
        #     f"{k}_{key}": value
        #     for k in self._env.agents
        #     for key, value in infos[k].items()
        # }
        if done:
            # empty obs and rewards for PZ environments on terminated episode
            assert len(obs) == 0
            assert len(rewards) == 0
            obs = self.last_obs
            rewards = [0] * len(obs)
        else:
            self.last_obs = obs
        return obs, rewards, done, truncated, infos

    def close(self):
        return self._env.close()

    def seed(self, seed=None):
        return self._env.seed(seed)
    @property
    def agents(self):
        return self._env.agents
  
    
    @property
    def action_spaces(self):
        return self._env.action_spaces

    @property
    def observation_spaces(self):
        return self._env.observation_spaces 
    
    def get_avail_agent_actions(self, agent_id):
        """返回agent_id对应的可用动作索引列表"""
        # 获取agent的ID
        agent = self._env.possible_agents[agent_id]#..
        # 直接返回底层环境中的可用动作索引列表
        return self._env.get_avail_agent_actions(agent)
    


# 注册所有 PettingZoo 环境
gym.register(
    "test-parallel-v4",
    entry_point="envs.test_wrapper1:EVPZWrapper",
    kwargs={
        "net_file": "src/dataset/cosmos/cosmos.net.xml",
        "sim_file": "src/dataset/cosmos/cosmos.sumocfg",
        "rou_file": "src/dataset/cosmos/cosmos.rou.xml",
        "add_file": "src/dataset/cosmos/cosmos.cs.add.xml",
        "begin_time": 0,
        "time_limit": 1250,
        "enable_gui": False,
        "sumo_seed": "random",
        "sumo_warnings": True,
        "virtual_display": (3200, 1800),
        "max_depart_delay": -1,
        "waiting_time_memory": 999999,
        "time_to_teleport": -1,
        "delta_time": 5,
        "render_mode": None,
        "additional_sumo_cmd": None,
        "output_file": "src/results/EV_results01"
        
    },
) 
