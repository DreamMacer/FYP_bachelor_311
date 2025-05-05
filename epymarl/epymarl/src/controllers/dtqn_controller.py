from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th
import torch.nn as nn
from utils.dtqn_utils import DTQNUtils

# This multi-agent controller shares parameters between agents
class DtqnMAC:
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.args = args
        self.scheme = scheme
        self.action_dim = args.action_embed_dim  
        self.context_len = args.context_len
        self.inner_embed_size = args.inner_embed_size 
        input_shape = self._get_input_shape(scheme)
        self._build_agents(input_shape)
        self.obs_dim = scheme["obs"]["vshape"]         
        self.agent_output_type = args.agent_output_type
        self.action_selector = action_REGISTRY[args.action_selector](args)
        self.device = args.device
        self.hidden_states = None
        

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # 这个函数在transmix-main/src/learners/q_learner.py中的train()函数中被调用
        # 用于在训练过程中选择每个智能体的动作
        #t_ep，当前时间步的可选择动作

        
        # history_batch = ep_batch[:, start_time_step:t_ep + 1]  # 使用截断的ep_batch
        # # 对transition_data中的每个key(obs,actions等)都只保留到t_ep的数据
        # history_batch.data.transition_data = {
        #     k: v[:, start_time_step:t_ep + 1] for k, v in history_batch.data.transition_data.items()
        # }  # ep_batch 包含了max_seq_length个时间步
        
        # 通过forward()获取每个智能体的Q值或策略
        avail_actions = ep_batch["avail_actions"][:, t_ep]#提取出在时间步 t_ep 上所有智能体的可用动作。
        agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)#[batch_size, context_len, n_agents, n_actions],获取最后context_len个Q
        agent_outputs = agent_outputs[:, -1, :, :]
        
        # 使用action_selector根据epsilon-greedy策略选择动作
        # bs是batch的切片索引,用于在训练或测试时选择特定的batch样本:
        # 1. 在训练时,通常使用整个batch进行训练,此时bs=slice(None)表示选择所有样本
        # 2. 在测试时,可能只需要评估部分样本,此时可以传入具体的索引值
        # 3. agent_outputs和avail_actions的形状都是[batch_size, n_agents, n_actions]
        # 4. 使用bs进行索引可以灵活地选择需要的batch样本进行动作选择
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)
        return chosen_actions

    def forward(self, ep_batch, t, test_mode=False):
        #被select_action调用时，t代表着一个episode中的当前时间步 
        inputs= self._build_inputs(ep_batch, t).to(self.device)#1,1,5,133
        # assert 1==0, f"inputs:{inputs.shape}"
        avail_actions = ep_batch["avail_actions"][:, t].to(self.device)#1,5,15
        if self.args.agent in ["rnn", "rnn2"]:
            agent_outs, self.hidden_states = self.agent(inputs, self.hidden_states)
            #print("ags: {}, hid: {}".format(agent_outs.shape, self.hidden_states.shape))
        elif self.args.agent in ["DTQNAgent"]:
            agent_outs = self.agent(inputs,self.args)#[bs,context_len, n_agents, input_shape]
        else:
            agent_outs = self.agent(inputs)
        #print("agent_outs: {}".format(agent_outs.shape))
        
        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":

            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10

            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)
            if not test_mode:
                # Epsilon floor
                epsilon_action_num = agent_outs.size(-1)
                if getattr(self.args, "mask_before_softmax", True):
                    # With probability epsilon, we will pick an available action uniformly
                    epsilon_action_num = reshaped_avail_actions.sum(dim=1, keepdim=True).float()

                agent_outs = ((1 - self.action_selector.epsilon) * agent_outs
                               + th.ones_like(agent_outs) * self.action_selector.epsilon/epsilon_action_num)

                if getattr(self.args, "mask_before_softmax", True):
                    # Zero out the unavailable actions
                    agent_outs[reshaped_avail_actions == 0] = 0.0

        return agent_outs

    def init_hidden(self, batch_size):
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav
        #self.agent.init_hidden()

    def parameters(self):
        return self.agent.parameters()

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def cuda(self):
        self.agent.cuda()

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))

    def _build_agents(self,input_shape):
        
        self.agent = agent_REGISTRY[self.args.agent](input_shape,self.args)

    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        #设置context_len
        if t < self.context_len:
            start_time_step = 0  # 当前时间步小于 context_len，从 0 开始
            context_len =t + 1
        else:
            start_time_step = t - self.context_len + 1  # 确保不超出范围
            context_len = self.context_len
        # assert 1==0, f"start_time_step:{start_time_step},type:{type(start_time_step)},t:{t}"
        #context_len,截取的长度
        self.dtqn_utils = DTQNUtils(obs_dim = self.obs_dim, 
                                    num_actions = self.n_actions,
                                    action_dim = self.action_dim,
                                    inner_embed_size=self.inner_embed_size,
                                    )
        #embedding 初始化
        if self.action_dim > 0:
            self.action_embedding = self.dtqn_utils.action_embedding
        else:
            self.action_embedding = None
        
        self.obs_embedding = self.dtqn_utils.obs_embedding
        bs = batch.batch_size
        #batch包含context_len个
        inputs = []
        # 获取从0到t的所有obs
        obs_inputs = batch["obs"][:, start_time_step:t + 1]#1，1，5，80

        # obs_embedding = self.obs_embedding(obs_inputs)

        # inputs.append(obs_embedding)  # bs, (context_len), n_agents, obs_dim
        
        if self.args.obs_last_action:#last_action :bs,context_len,n_agents,1
            last_actions = []
            if t ==0:
                last_actions.append(th.zeros_like(batch["actions_onehot"][:, 0]).unsqueeze(1))# 0时刻的last_action用0填充
            elif t < self.context_len:
                last_actions.append(th.zeros_like(batch["actions_onehot"][:, 0]).unsqueeze(1))
                last_actions.append(batch["actions_onehot"][:, start_time_step:t])       
            else:
                last_actions.append(batch["actions_onehot"][:, start_time_step-1:t]) #第一次改为actions # bs, context_len, n_agents, action_dim
            last_actions = th.cat(last_actions, dim=1)  # bs, (context_len), n_agents, action_dim
            last_actions = last_actions.long() 
            # if self.action_embedding is not None:
            #     action_embedding = self.action_embedding(last_actions)
            #     inputs.append(action_embedding) #action_embedding = None
            if self.args.obs_agent_id:
            # 扩展agent id到所有时间步
                agent_ids = th.eye(self.n_agents, device=batch.device).unsqueeze(0).unsqueeze(0)
                agent_ids = agent_ids.expand(bs,context_len, -1, -1)
        # if self.action_embedding is not None and self.args.obs_agent_id:
        #     inputs = th.cat([obs_embedding, action_embedding, agent_ids], dim=-1)
        # elif self.action_embedding is not None:
        #     inputs = th.cat([obs_embedding, action_embedding], dim=-1)
        # elif self.args.obs_agent_id:
        #     inputs = th.cat([obs_embedding, agent_ids], dim=-1)
        # else:
        #     inputs = obs_embedding#inner_embed_size+n_agents
        # # assert 1==0, f"obs:{obs_embedding.shape},last_actions:{action_embedding.shape},agent_id:{agent_ids.shape},inputs : {inputs.shape}"
        # # 最终inputs的形状为: [bs, context_len, n_agents, input_shape]
        # # 其中input_shape = embed_obs_dim + embed_action_dim(如果使用last_action) + n_agents(如果使用agent_id)  
        # # assert inputs.shape[-1] ==  self.inner_embed_size, f"Expected last dimension to be {self.inner_embed_size}, but got {inputs.shape[-1]}"           
        # # assert 1==0,f"inputs : {inputs.shape}"
        #--------------simplify---------------
        # assert 1==0, f"obs:{obs_inputs.shape},last_actions:{last_actions.shape},agent_id:{agent_ids.shape}"
        inputs =th.cat([obs_inputs,last_actions,agent_ids],dim=-1)
        # assert 1==0, f"inputs,{inputs.shape}"#1.1.5.86
        return inputs
    

    def _get_input_shape(self, scheme):#未嵌入时的shape
        # input_shape = self.inner_embed_size+self.n_agents
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape
