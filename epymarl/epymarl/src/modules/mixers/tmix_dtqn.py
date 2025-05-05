
from numpy.core.fromnumeric import shape
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from utils import torch_utils

class DMixer(nn.Module):
    def __init__(self, scheme, input_shape, args):
        super(DMixer, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.scheme = scheme
        
        self.input_shape = input_shape
        self.state_dim = int(np.prod(args.state_shape))
        self.n_actions = args.n_actions
        self.action_dim = args.n_agents * self.n_actions
        self.state_action_dim = self.state_dim + self.action_dim + 1
        self.max_seq_len = args.batch_size * args.eps_limit
        self.embed_dim = args.embed_dim
        self.obs_dim = self.n_agents * self.scheme['obs']['vshape']
        #self.hist_dim = self.args.n_agents * self.args.rnn_hidden_dim
        #self.obs_dim = int(np.prod(args.obs_shape))
        dim = args.embed_dim * 3
        
        self.state_transform = nn.Linear(self.state_dim, args.embed_dim)#value transformation        
        self.aqs_transform = nn.Linear(self.n_agents, args.embed_dim)#key transformation        
        self.obs_transform = nn.Linear(self.obs_dim, args.embed_dim)#query transformation            
        
        self.enc_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=args.heads, dim_feedforward=args.ff, dropout=0.4, 
                                                    activation='relu')
        self.mixer = nn.TransformerEncoder(self.enc_layer, num_layers=args.t_depth)
        self.bottleneck = nn.Linear(dim, dim)
        self.out = nn.Sequential( 
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, 1)
            
        )
        self.apply(torch_utils.init_weights)
    def create_noise(self, states, mean=0, stddev=0.05):
        noise = th.as_tensor(states, dtype=th.float).normal_(mean, stddev).cuda()
        return noise
    
    def calc_v(self, agent_qs):
        v_tot = th.sum(agent_qs, dim=-1, keepdim=True)
        return v_tot

    def forward(self, agent_qs, obs, states, b_max=0):  # obs_tensor形状: [batch_size, max_seq_length, n_agents, obs_dim]
        v = self.calc_v(agent_qs)

        if self.args.is_noise == True:
            noise = self.create_noise(states)
            states = ((noise + states).detach() - states).detach() + states

        states = th.abs(self.state_transform(states)).to(agent_qs.device)
        agent_qs = self.aqs_transform(agent_qs).to(agent_qs.device)
        #obs = th.abs(self.obs_transform(obs)).to(agent_qs.device)
        
        obs = obs.contiguous().view(obs.shape[0], obs.shape[1], -1)
        
        obs = self.obs_transform(obs).to(agent_qs.device)
        
        # This one is based on the vanilla transformer 
        x = th.cat([states, agent_qs, agent_qs], dim=2)#obs
        x = self.bottleneck(x)
        q = self.mixer(x) + x
        q_tot = (self.out(q) * x) + v
        del x, q, v
        
        return q_tot


        
