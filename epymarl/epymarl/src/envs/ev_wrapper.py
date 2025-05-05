import os
import sys
import gymnasium as gym
from pettingzoo.utils.conversions import aec_to_parallel
from pettingzoo.utils import wrappers
from typing import Optional, Dict, Any
from .pz_wrapper import PettingZooWrapper

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)

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
    # from .ev_env import EVPZ # 导入原始环境
    from .ev_parallel import EVParallelWrapper  # 导入原始环境
    # 创建基础环境
    env = EVParallelWrapper(**kwargs)
    
    # # 应用 PettingZoo 包装器
    # env = wrappers.AssertOutOfBoundsWrapper(env)
    # env = wrappers.OrderEnforcingWrapper(env)
    # # CaptureStdoutWrapper 应该在最后应用
    # if kwargs.get("render_mode") == "human":
    #     env = wrappers.CaptureStdoutWrapper(env)
    
    return env

def parallel_env(**kwargs):
    """创建并返回 Parallel 版本的 SUMO 环境"""
    env = make_env(**kwargs)  # 先创建 AEC 环境实例
    # env = aec_to_parallel(env)  # 转换为 Parallel 环境
    env = PettingZooWrapper(env, env_name="ev-parallel-v2")  # 使用 PettingZooWrapper 包装，添加环境名称
    return env

# 注册环境
gym.register(
    id="ev-parallel-v2",
    entry_point="envs.ev_wrapper:parallel_env",
    kwargs={
        "net_file": "src/dataset/cosmos/cosmos.net.xml",
        "sim_file": "src/dataset/cosmos/cosmos.sumocfg",
        "rou_file": "src/dataset/cosmos/cosmos.rou.xml",
        "add_file": "src/dataset/cosmos/cosmos.cs.add.xml",
        "begin_time": 0,
        "seconds": 10000,
        "enable_gui": False,
        "sumo_seed": "1",
        "sumo_warnings": True,
        "virtual_display": (3200, 1800),
        "max_depart_delay": -1,
        "waiting_time_memory": 1000,
        "time_to_teleport": -1,
        "delta_time": 5,
        "render_mode": None,
        "additional_sumo_cmd": None,
        "output_file": "src/results/EV_results01"
        
    },
) 