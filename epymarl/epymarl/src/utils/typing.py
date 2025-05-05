# from typing import Type, Tuple, Any, List
# import numpy as np
# import numpy.typing as npt
# import gymnasium as gym
# from gym.spaces import Space


# VertexType = Tuple[float]
# VerticesType = npt.NDArray[VertexType]
# DemandType = npt.NDArray[float]
# EdgeType = npt.NDArray[Tuple[int]]
# DeparturesType = npt.NDArray[int]
# CapacityType = npt.NDArray[float]
# AdjListType = npt.NDArray[npt.NDArray[int]]
# LocationsType = npt.NDArray[int]
# LoadingType = npt.NDArray[Tuple[float]]
# SpaceType = Space
# ActionsType = npt.NDArray[int]
# RewardsType = npt.NDArray[float]

from typing import Tuple, Any
import numpy as np
import numpy.typing as npt
from gym.spaces import Space

# 基础类型定义
VertexType = Tuple[float, ...]  # 允许多个 float
VerticesType = npt.NDArray[np.float64]  # 顶点坐标通常是 float64 类型

DemandType = npt.NDArray[np.float64]  # 需求值通常是浮点数
EdgeType = npt.NDArray[np.int64]  # 边索引通常是整数
DeparturesType = npt.NDArray[np.int64]  # 出发时间/索引通常是整数
CapacityType = npt.NDArray[np.float64]  # 负载/容量通常是浮点数

# 复杂数据结构
AdjListType = npt.NDArray[np.object_]  # Adjacency List 存储多个 NumPy 数组
LocationsType = npt.NDArray[np.int64]  # 位置索引通常是整数
LoadingType = npt.NDArray[np.float64]  # 负载情况存储浮点数
ActionsType = npt.NDArray[np.int64]  # 动作空间通常是整数
RewardsType = npt.NDArray[np.float64]  # 奖励通常是浮点数

# Gym 相关类型
SpaceType = Space