import torch as th
import torch.nn as nn
from utils.gates import GRUGate, ResGate
#from modules.networks.transformer import TransformerLayer, TransformerIdentityLayer
from utils import torch_utils
from numpy.core.fromnumeric import shape
import torch.nn.functional as F
import numpy as np
from einops import rearrange

class DTQN(nn.Module):
    def __init__(self, scheme, input_shape, args):
        super(DTQN, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.scheme = scheme
        
        self.input_shape = input_shape
        self.state_dim = int(np.prod(args.state_shape))
        self.n_actions = args.n_actions
        self.action_dim = args.n_agents * self.n_actions
        self.state_action_dim = self.state_dim + self.action_dim + 1
        self.max_seq_len = args.batch_size * args.eps_limit#
        self.embed_dim = args.embed_dim
        self.obs_dim = self.n_agents * self.scheme['obs']['vshape']
        self.hist_dim = self.args.n_agents * self.args.rnn_hidden_dim
        dropout = args.dtqn_dropout
        self.state_transform = nn.Linear(self.state_dim, args.embed_dim)#state transformation        
        self.aqs_transform = nn.Linear(self.n_agents, args.embed_dim)#key transformation        
        self.hist_transform = nn.Linear(self.hist_dim, args.embed_dim)#value transformation
        self.dropout = nn.Dropout(dropout)
        self.num_heads = args.num_heads #
        self.gate = args.gate
        
        if self.gate == "gru":
            self.attn_gate = GRUGate(embed_size=self.embed_dim)
            self.mlp_gate = GRUGate(embed_size=self.embed_dim)
        elif self.gate == "res":
            self.attn_gate = ResGate()
            self.mlp_gate = ResGate()
        else:
            raise ValueError("Gate must be one of `gru`, `res`")

        # if self.identity:
        #     transformer_block = TransformerIdentityLayer
        # else:
        #     transformer_block = TransformerLayer
        # self.transformer_layers = nn.Sequential(
        #     *[
        #         transformer_block(
        #             self.num_heads,
        #             self.embed_dim,
        #             self.max_seq_len,
        #             dropout,
        #             self.attn_gate,
        #             self.mlp_gate,
        #         )
        #         for _ in range(self.num_layers)
        #     ]
        # )

        self.ffn = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.n_actions),
        )
        self.ffn2 = nn.Sequential(
            nn.Linear(self.embed_dim, 4 * self.embed_dim),
            nn.ReLU(),
            nn.Linear(4 * self.embed_dim, self.embed_dim),
            nn.Dropout(dropout),
        )
        
        self.attention = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.alpha = None
        self.attn_mask = nn.Parameter(
            th.triu(th.ones(self.max_seq_len-1, self.max_seq_len-1), diagonal=1),
            requires_grad=False,
        )
        self.attn_mask[self.attn_mask.bool()] = -float("inf")
        
        self.dropout = nn.Dropout(dropout)
        self.apply(torch_utils.init_weights)
        self.layernorm1 = nn.LayerNorm(self.embed_dim)
        self.layernorm2 = nn.LayerNorm(self.embed_dim)    
        self.linear = nn.Linear(2*self.embed_dim,self.embed_dim) 
        self.out = nn.Sequential( 
            nn.Linear(self.embed_dim, self.embed_dim // 2),
            nn.GELU(),
            nn.Linear(self.embed_dim // 2, self.embed_dim // 4),
            nn.GELU(),
            nn.Linear(self.embed_dim // 4, 1)
            
        )      
        
    def create_noise(self, states, mean=0, stddev=0.05):
        noise = th.as_tensor(states, dtype=th.float).normal_(mean, stddev).cuda()
        return noise

    def forward(self, agent_qs, hist, states, b_max=0):  #hist即在mac（调用rnnagent)产生的hidden state
        # chosen_action_qvals: [batch_size, max_seq_length-1, n_agents] 表示每个智能体选择的动作的Q值
        # mac_hidden_states: [batch_size, max_seq_length-1, n_agents, hidden_dim] 表示RNN的隐藏状态
        # batch["state"]: [batch_size, max_seq_length-1, state_dim] 表示环境状态
        if self.args.is_noise == True:
            noise = self.create_noise(states)
            states = ((noise + states).detach() - states).detach() + states
        #----------------Transformation----------bs,seq-1,embed
        #------------encoder-part---------
        states = th.abs(self.state_transform(states)).to(agent_qs.device)
        agent_qs = self.aqs_transform(agent_qs).to(agent_qs.device)
        hist = hist.contiguous().view(hist.shape[0], hist.shape[1], -1)
        hist = self.hist_transform(hist).to(agent_qs.device)
        
        q_k = th.cat((states,hist),dim=2)#bs,seq-1.2*embed,64,64,1024
        q_k_trans = self.linear(q_k)#64,64,512
        q_k_att, self.alpha = self.attention(q_k_trans, q_k_trans, q_k_trans)#64,64,512
        q_k_att = self.attn_gate(q_k_att,F.relu(q_k_trans))
        q_k_att = self.layernorm1(q_k_att)
        ffn2 = self.ffn2(q_k_att)
        q_k_out = self.mlp_gate(q_k_att,F.relu(ffn2))#64,64,512
        #-------------decoder-part------------
        attention, self.alpha = self.attention(
            agent_qs,#value
            agent_qs,#key
            agent_qs,#query
            attn_mask=self.attn_mask[: agent_qs.size(1), : agent_qs.size(1)],#self.attn_mask 截取为一个大小为 seq_len x seq_len 的子矩阵
        )#64,64,512
        x = self.attn_gate(agent_qs, F.relu(attention))#64,64,512
        v = self.layernorm1(x)#64,64,512
        q_k_v,self.alpha = self.attention(q_k_out,q_k_out,v)
        x = self.attn_gate(v, F.relu(q_k_v))
        x = self.layernorm1(x)
        ffn= self.ffn2(x)
        x = self.mlp_gate(x, F.relu(ffn))
        x = self.layernorm2(x)
        q_tot= self.out(x)
        return q_tot
        
        
        

        

        

    