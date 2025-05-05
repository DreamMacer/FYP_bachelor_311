import os
import sys
import gymnasium as gym
from pettingzoo.utils.conversions import aec_to_parallel
from pettingzoo.utils import wrappers
from typing import Optional, Dict, Any
from .pz_wrapper import PettingZooWrapper

# 确保 SUMO_HOME 环境变量已设置
if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    raise ImportError("请先设置环境变量 'SUMO_HOME'")

# 设置环境变量以避免重复库加载
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def make_env(**kwargs) -> Any:
    """创建并包装环境"""
    from .sumo_env import SumoEVEnvironmentPZ  # 导入原始环境
    
    # 创建基础环境
    env = SumoEVEnvironmentPZ(**kwargs)
    
    # 应用 PettingZoo 包装器
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    # CaptureStdoutWrapper 应该在最后应用
    if kwargs.get("render_mode") == "human":
        env = wrappers.CaptureStdoutWrapper(env)
    
    return env

def parallel_env(**kwargs):
    """创建并返回 Parallel 版本的 SUMO 环境"""
    env = make_env(**kwargs)  # 先创建 AEC 环境实例
    env = aec_to_parallel(env)  # 转换为 Parallel 环境
    env = PettingZooWrapper(env, env_name="sumo-ev-parallel-v0")  # 使用 PettingZooWrapper 包装，添加环境名称
    return env

# 注册环境
gym.register(
    "sumo-ev-parallel-v0",
    entry_point="envs.sumo_wrapper:parallel_env",
    kwargs={
        "net_file": "src/dataset/4_station_strip/4_station_strip.net.xml",
        "sim_file": "src/dataset/4_station_strip/4_station_strip.sumocfg",
        "begin_time": 0,
        "seconds": 10000,
        "enable_gui": False,
        "sumo_seed": "random",
        "sumo_warnings": True,
        "add_per_agent_info": True,
        "virtual_display": (3200, 1800),
        "max_depart_delay": -1,
        "waiting_time_memory": 1000,
        "time_to_teleport": -1,
        "delta_time": 5,
        "add_system_info": True,
        "additional_sumo_cmd": None,
        "output_file": "src/results/sumo_results01",
        "render_mode": None,
    },
)
