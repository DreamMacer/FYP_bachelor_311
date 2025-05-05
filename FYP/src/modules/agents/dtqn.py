import torch
import torch.nn as nn
from modules.networks.position_encodings import PosEnum, PositionEncoding
from modules.networks.gates import GRUGate, ResGate
#from modules.networks.transformer import TransformerLayer, TransformerIdentityLayer
from modules.networks.transformer2 import TransformerLayer, TransformerIdentityLayer
from utils import torch_utils


class DTQN(nn.Module):
    """Deep Transformer Q-Network for partially observable reinforcement learning.

    Args:
        obs_dim:            The length of the observation vector.
        num_actions:        The number of possible environments actions.
        embed_per_obs_dim:  Used for discrete observation space. Length of the embed for each
            element in the observation dimension.
        action_dim:         The number of features to give the action.
        inner_embed_size:   The dimensionality of the network. Referred to as d_k by the
            original transformer.
        num_heads:          The number of heads to use in the MultiHeadAttention.
        num_layers:         The number of transformer blocks to use.
        history_len:        The maximum number of observations to take in.
        dropout:            Dropout percentage. Default: `0.0`
        gate:               Which layer to use after the attention and feedforward submodules (choices: `res`
            or `gru`). Default: `res`
        identity:           Whether or not to use identity map reordering. Default: `False`
        pos:                The kind of position encodings to use. `0` uses no position encodings, `1` uses
            learned position encodings, and `sin` uses sinusoidal encodings. Default: `1`
        discrete:           Whether or not the environment has discrete observations. Default: `False`
        vocab_sizes:        If discrete env only. Represents the number of observations in the
            environment. If the environment has multiple obs dims with different number
            of observations in each dim, this can be supplied as a vector. Default: `None`
        input_shape: obs_dim + action_dim(如果使用last_action) + n_agents(如果使用agent_id)
    """

    def __init__(self, input_shape, args):
        super(DTQN, self).__init__()
        self.input_shape = input_shape
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.action_dim = args.action_embed_dim
        self.max_seq_len = args.batch_size * args.eps_limit
        self.inner_embed_size = args.inner_embed_size
        self.num_heads = args.num_heads
        self.num_layers = args.num_layers
        dropout = args.dtqn_dropout
        self.gate = args.gate
        self.identity = args.identity   
        self.pos = args.pos
        self.discrete = args.discrete
        self.vocab_sizes = args.vocab_sizes
        self.history_len = args.eps_limit + 1
        self.embed_dim = args.embed_dim
        # 添加 device 参数
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        assert self.inner_embed_size == 128, f"Expected inner_embed_size to be 128, but got {self.inner_embed_size}"
        self.transform = nn.Linear(self.input_shape, self.embed_dim)
        # Input Embedding: Allocate space for the action embedding
        pos_function_map = {
            PosEnum.LEARNED: PositionEncoding.make_learned_position_encoding,
            PosEnum.SIN: PositionEncoding.make_sinusoidal_position_encoding,
            PosEnum.NONE: PositionEncoding.make_empty_position_encoding,
        }
        self.position_embedding = pos_function_map[PosEnum(self.pos)](
            context_len=self.history_len, embed_dim=self.embed_dim, n_agent=self.n_agents
        )#产生长度为episode_lenth的嵌入，不是context_len
        
        #self.transform = nn.Linear(self.obs_input_shape, self.inner_embed_size)
        self.dropout = nn.Dropout(dropout)

        if self.gate == "gru":
            attn_gate = GRUGate(embed_size=self.embed_dim)
            mlp_gate = GRUGate(embed_size=self.embed_dim)
        elif self.gate == "res":
            attn_gate = ResGate()
            mlp_gate = ResGate()
        else:
            raise ValueError("Gate must be one of `gru`, `res`")

        if self.identity:
            transformer_block = TransformerIdentityLayer
        else:
            transformer_block = TransformerLayer
        self.transformer_layers = nn.Sequential(
            *[
                transformer_block(
                    self.num_heads,
                    self.embed_dim,
                    self.history_len,
                    dropout,
                    attn_gate,
                    mlp_gate,
                )
                for _ in range(self.num_layers)
            ]
        )

        self.ffn = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.n_actions),
        )
        
        
        self.apply(torch_utils.init_weights)

    def forward(self, inputs: torch.Tensor,t) -> torch.Tensor:
        """       
        输入:
        -inputs: 观察序列张量 [batch_size, n_agents, context_len, embed_obs_dim + embed_action_dim + n_agents]
        输出:
        - last_step_q_values: 最后一个时间步的Q值 [batch_size, n_agents, n_actions]
        """
        inputs = inputs.to(self.device)
        batch_size = inputs.shape[0]
        context_len = inputs.shape[1]
   
   
        assert context_len <= self.history_len, "Cannot forward, history is longer than expected.{context_len}shape:{context_len.shape}"

        # 3. 位置编码: 添加序列位置信息
        # 输入: n_agents, history_len
        # 输出: [1, seq_len, n_agents, embed_dim] -> [bs, context_len, n_agents, embed_dim]
        pos_encoding = self.position_embedding.forward()
        pos_encoding = pos_encoding.unsqueeze(0).expand(batch_size, -1, -1, -1)[:, :context_len, :, :]
        # token_embeddings = inputs.view(self.batch_size, context_len, n_agents, -1)
        
        # 5. Transformer层: 处理序列信息
        # assert 1==0,f"inputs : {inputs.shape},pos={pos_encoding.shape}"
        inputs=self.transform(inputs)#256
        working_memory = inputs + pos_encoding
        # assert 1==0,f"working_memory : {working_memory.shape}，context_len={context_len},bs={batch_size},inputs={inputs.shape},pos={pos_encoding.shape}"
        working_memory = self.transformer_layers(
            self.dropout(working_memory.view(batch_size,self.n_agents*context_len,-1))
        )
        output = self.ffn(working_memory.view(-1, self.embed_dim))
        # assert 1==0,f"output : {output.shape},bs={batch_size}"
        
        # 8. 提取最后一个时间步的Q值
        output = output.view(batch_size, context_len, self.n_agents, -1)

        return output[:, -context_len:, :, :]#[batch_size, context_len, n_agents, n_actions],获取最后context_len个Q
