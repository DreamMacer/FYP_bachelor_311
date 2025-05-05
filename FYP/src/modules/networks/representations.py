from __future__ import annotations
from typing import Tuple, Optional
import math
from utils import torch_utils
import torch
import torch.nn as nn


class ObservationEmbeddingRepresentation(nn.Module):
    def __init__(
        self,
        observation_embedding: nn.Module,
    ):
        super().__init__()
        self.observation_embedding = observation_embedding
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        self.apply(torch_utils.init_weights)

    def forward(self, obs: torch.Tensor):
        # obs: [batch_size, context_len, n_agents, obs_dim]
        if obs.is_cuda:  # 检查输入是否在 GPU 上
            obs = obs.to(torch.float32)  # 确保数据类型为 FloatTensor
        batch, context_len, n_agents,  features = obs.size()
        # Flatten batch, n_agents, and seq dims
        obs = obs.reshape(batch * context_len * n_agents , features)
        obs_embed = self.observation_embedding(obs)
        # assert 1==0,f"obs:{obs.shape},obs_embed,{obs_embed.shape}，context_len,{context_len}"
        # Reshape back to [batch_size, seq_len, n_agents, embedding_size]
        obs_embed = obs_embed.reshape(batch, context_len, n_agents, -1)
        return obs_embed
    #将输入的观察数据 obs 通过一个嵌入层进行转换，并将结果重新调整形状以恢复原始的批次和时间步维度

    @staticmethod
    def make_discrete_representation(
        vocab_sizes: int, obs_dim: int, embed_per_obs_dim: int, outer_embed_size: int
    ) -> ObservationEmbeddingRepresentation:
        """
        For use in discrete observation environments.

        Args:
            vocab_sizes:        The number of different values your observation could include.
            obs_dim:            The length of the observation vector (assuming 1d).
            embed_per_obs_dim:  The number of features you want to give to each observation
                dimension.
            embed_size:         The length of the resulting embedding vector.
        """

        assert (
            vocab_sizes > 0
        ), "Discrete environments need to have a vocab size for the token embeddings"
        assert (
            embed_per_obs_dim > 1
        ), "Each observation feature needs at least 1 embed dim"

        embedding = nn.Sequential(
            nn.Embedding(vocab_sizes, embed_per_obs_dim),
            #是一个用于查找表的层，它将数量为vocab_sizes的observations映射到embed_per_obs_dim维嵌入空间
            nn.Flatten(start_dim=-2),
            # 会展平多维张量。在这里，它将从倒数第二维（-2）开始展平
            nn.Linear(embed_per_obs_dim * obs_dim, outer_embed_size),
            #输入大小是 embed_per_obs_dim * obs_dim，即展平后的输出维度，输出大小是 outer_embed_size，表示经过该线性变换后的嵌入空间的维度
        )
        return ObservationEmbeddingRepresentation(observation_embedding=embedding)

    @staticmethod
    def make_action_representation(
        num_actions: int,
        action_dim: int,
    ) -> ObservationEmbeddingRepresentation:
        embed = nn.Sequential(
            nn.Embedding(num_actions, action_dim), nn.Flatten(start_dim=-2)
        )
        return ObservationEmbeddingRepresentation(observation_embedding=embed)

    @staticmethod
    def make_continuous_representation(obs_dim: int, outer_embed_size: int):
        """
        For use in continuous observation environments. Projects the observation to the
            specified dimensionality for use in the network.

        Args:
            obs_dim:    The length of the observation vector (assuming 1d)
            embed_size: The length of the resulting embedding vector
        """
        embedding = nn.Linear(obs_dim, outer_embed_size)
        return ObservationEmbeddingRepresentation(observation_embedding=embedding)

    @staticmethod
    def make_image_representation(obs_dim: Tuple, outer_embed_size: int):
        """
        For use in image observatino environments.

        Args:
            obs_dim:            The image observation's dimensions (C x H x W).
            outer_embed_size:   The length of the resulting embedding vector.
        """
        # C x H x W or H x W
        #检查图像维度 obs_dim 的长度。如果是三维数据（C x H x W），则 num_channels 设置为 obs_dim[0]（即通道数）。否则，默认为灰度图像，通道数为 1
        if len(obs_dim) == 3:
            num_channels = obs_dim[0]
        else:
            num_channels = 1

        kernels = [3, 3, 3, 3, 3]
        paddings = [1, 1, 1, 1, 1]
        strides = [2, 1, 2, 1, 2]
        flattened_size = compute_flattened_size(
            obs_dim[1], obs_dim[2], kernels, paddings, strides
        )
        embedding = nn.Sequential(
            # Input 3 x 84 x 84
            nn.Conv2d(
                num_channels,#input:3 layers
                64,#output:64 layers
                kernel_size=kernels[0],#3
                padding=paddings[0],#1
                stride=strides[0],#2
            ),
            nn.ReLU(True),
            #
            nn.Conv2d(
                64, 64, kernel_size=kernels[1], padding=paddings[1], stride=strides[1]
            ),
            nn.ReLU(True),
            nn.Conv2d(
                64,
                64,
                kernel_size=kernels[2],
                padding=paddings[2],
                stride=strides[2],
            ),
            nn.ReLU(True),
            nn.Conv2d(
                64, 128, kernel_size=kernels[3], padding=paddings[3], stride=strides[3]
            ),
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(128 * flattened_size, outer_embed_size),
        )
        return ObservationEmbeddingRepresentation(observation_embedding=embedding)


def compute_flattened_size(
    height: int, width: int, kernels: list, paddings: list, strides: list
) -> int:
    #计算的结果是图像的总元素数量，也就是卷积神经网络（CNN）处理图像后展平层的输入大小
    for i in range(len(kernels)):
        height = update_size(height, kernels[i], paddings[i], strides[i])
        width = update_size(width, kernels[i], paddings[i], strides[i])
    return int(height * width)


def update_size(component: int, kernel: int, padding: int, stride: int) -> int:
    return math.floor((component - kernel + 2 * padding) / stride) + 1


# class ActionEmbeddingRepresentation(nn.Module):
#     def __init__(self, num_actions: int, action_dim: int):
#         super().__init__()
#         self.embedding = nn.Sequential(
#             nn.Embedding(num_actions, action_dim),#num_actions=avail_actions,action_dim=dimensions to represent an action
#             nn.Flatten(start_dim=-2),
#         )

#     def forward(self, action: torch.Tensor):
#         # action: [batch_size, seq_len, n_agents, n_actions]
#         if action.is_cuda:  # 检查输入是否在 GPU 上
#             action = action.to(torch.float32)  # 确保数据类型为 FloatTensor
#         batch, seq, n_agents, features = action.size()
#         # Flatten batch, n_agents, and seq dims
#         action = action.reshape(batch * seq * n_agents, features)
#         action_embed = self.embedding(action.long())
#         # Reshape back to [batch_size, n_agents, seq_len, embedding_size]
#         action_embed = action_embed.reshape(batch, seq, n_agents, -1)
#         return action_embed
class ActionEmbeddingRepresentation(nn.Module):
    def __init__(self, num_actions: int, action_dim: int):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Embedding(num_actions, action_dim),
            nn.Flatten(start_dim=-2),
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        self.apply(torch_utils.init_weights)
    def forward(self, action: torch.Tensor):
        return self.embedding(action)