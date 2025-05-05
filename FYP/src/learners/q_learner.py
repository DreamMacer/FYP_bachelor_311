import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
from modules.mixers.tmix_vanilla import TMixer
import torch as th
import random
from torch.optim import RMSprop
from torch.optim import Adam
from torch.optim import AdamW
from torch.amp import GradScaler, autocast


class QLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger
        self.context_len = args.context_len
        
        self.params = list(mac.parameters())

        self.last_target_update_episode = 0

        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "vdn":
                self.mixer = VDNMixer()
            elif args.mixer == "qmix":
                self.mixer = QMixer(args)
            elif args.mixer == "tmix":
                self.input_shape = args.inner_embed_size + args.n_agents
                self.mixer = TMixer(scheme, input_shape=self.input_shape, args=self.args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        #self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha,weight_decay= 1e-4, eps=args.optim_eps)
        #self.optimiser = Adam(params=self.params, lr=args.lr) #, eps=args.optim_eps)
        self.optimiser = AdamW(params=self.params, lr=args.lr,weight_decay = 1e-4) #, eps=args.optim_eps)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1
        self.scaler = GradScaler()


    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
         # Get the relevant quantities
        rewards = batch["reward"]
        actions = batch["actions"]
        terminated = batch["terminated"].float()
        avail_actions = batch["avail_actions"]

        # Calculate estimated Q-Values
        mac_out = []
        target_mac_out = []
        # assert 1==0,f"batch.batch_size: {batch.batch_size},actions.shape: {actions.shape},avail_actions.shape: {avail_actions.shape}"    
        start_t = random.randint(0, max(0, batch.max_seq_length - self.context_len))          
        for episode in range(batch.batch_size):#????应该使用最小的episode长度#效率较低，有更高效的
            episode_batch = batch[episode:episode+1]
            # 随机选择起始位置
            agent_outs = self.mac.forward(episode_batch, start_t + self.context_len-1)#返回的本身就是张量，[batch_size=1, context_len, n_agents, n_actions]
            target_agent_outs = self.target_mac.forward(episode_batch, start_t + self.context_len-1)
            # print(f"agent_out shape: {agent_outs.shape}")  # 打印 mac_out 的形状
            # print(f"target_agent_out shape: {target_agent_outs.shape}")  # 打印 target_agent_out的形状       
            mac_out.append(agent_outs)  # 保存Q值,
            target_mac_out.append(target_agent_outs)#保存target-Q值
        avail_actions = avail_actions[:, start_t:start_t + self.context_len]#avail_actions的形状是[batch_size, context_len, n_agents, n_actions]
        # print(f"avail_actions shape: {avail_actions.shape}")  # 打印 avail_actions 的形状
        mac_out = th.cat(mac_out, dim=0)  # Concat over batch_size
        target_mac_out = th.cat(target_mac_out, dim=0)[:, 1:] #batchsize,context_len,n_agents,n_actions
        # print(f"mac_out shape2: {mac_out.shape}")  # 打印 mac_out 的形状
        # print(f"target_agent_out shape2: {target_agent_outs.shape}")  # 打印 target_agent_out的形状
        # print(f"avail shape: {avail_actions[:, 1:].shape}")  
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999
        
        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions[:, start_t:start_t + self.context_len-1]).squeeze(3)


        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, :-1].max(dim=3, keepdim=True)[1] 
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]

        # Mix
        if self.mixer is not None:
            
            if self.args.mixer == "qmix":
                chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, start_t:start_t + self.context_len-1])
                target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, start_t + 1:start_t + self.context_len ])
            elif self.args.mixer == "tmix":            
                chosen_action_qvals = self.mixer(chosen_action_qvals, batch["obs"][:, start_t:start_t + self.context_len-1], batch["state"][:, start_t:start_t + self.context_len-1])
                target_max_qvals = self.target_mixer(target_max_qvals, batch["obs"][:, start_t + 1:start_t + self.context_len], batch["state"][:, start_t + 1:start_t + self.context_len])
                # chosen_action_qvals = self.mixer(chosen_action_qvals, mac_hidden_states[:,:-1], batch["state"][:, :-1])
                # target_max_qvals = self.target_mixer(target_max_qvals, target_mac_hidden_states[:,1:], batch["state"][:, 1:])

        # Calculate 1-step Q-Learning targets
        targets = rewards[:, start_t:start_t + self.context_len-1] + self.args.gamma * (1 - terminated[:, start_t:start_t + self.context_len-1]) * target_max_qvals

        mask = batch["filled"][:, start_t:start_t + self.context_len-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, start_t:start_t + self.context_len-2])
        # Td-error
        with autocast('cuda'):
            td_error = (chosen_action_qvals - targets.detach())

            mask = mask.expand_as(td_error)

            # 0-out the targets that came from padded data
            masked_td_error = td_error * mask

            # Normal L2 loss, take mean over actual data
            loss = (masked_td_error ** 2).sum() / mask.sum()

        # Optimise
        # if hasattr(self, 'scaler'):
        #     self.optimiser.zero_grad()  # 清零梯度
        #     self.scaler.scale(loss).backward()  # 缩放损失并反向传播
        #     self.scaler.unscale_(self.optimiser)  # 反缩放梯度
        #     grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)  # 梯度裁剪
        #     self.scaler.step(self.optimiser)  # 更新优化器
        #     self.scaler.update()  # 更新 scaler 的状态
        # else:
            # 原始优化流程
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)

            self.log_stats_t = t_env

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
