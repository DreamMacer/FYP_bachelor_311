from __future__ import annotations
import torch
import torch.nn as nn
import numpy as np
from enum import Enum


class PosEnum(Enum):
    LEARNED = "learned"
    SIN = "sin"
    NONE = "none"


class PositionEncoding(nn.Module):
    def __init__(self, position_encoding: nn.Module):
        super().__init__()
        self.position_encoding = position_encoding

    def forward(self):
        return self.position_encoding

    @staticmethod
    def make_sinusoidal_position_encoding(
        context_len: int, embed_dim: int, n_agent: int
    ) -> PositionEncoding:
        position = torch.arange(context_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2) * (-np.log(10000.0) / embed_dim)
        )
        pos_encoding = torch.zeros(context_len, n_agent, embed_dim)
        pos_encoding[:, :, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, :, 1::2] = torch.cos(position * div_term)
        return PositionEncoding(nn.Parameter(pos_encoding, requires_grad=False))

    @staticmethod
    def make_learned_position_encoding(
        context_len: int, embed_dim: int, n_agent: int
    ) -> PositionEncoding:
        return PositionEncoding(
            nn.Parameter(torch.zeros(context_len, n_agent, embed_dim), requires_grad=True)
        )

    @staticmethod
    def make_empty_position_encoding(
        context_len: int, embed_dim: int, n_agent: int
    ) -> PositionEncoding:
        return PositionEncoding(
            nn.Parameter(torch.zeros(context_len, n_agent, embed_dim), requires_grad=False)
        )
