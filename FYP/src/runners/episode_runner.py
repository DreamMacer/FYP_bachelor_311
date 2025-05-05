from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
import numpy as np

class EpisodeRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = args.batch_size_run
        assert self.batch_size == 1 ,f"batchsize ={self.batch_size}"

        self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        self.episode_limit = self.env.episode_limit
        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        # Log the first run
        self.log_train_stats_t = -1000000
     
        
        
        
    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch = self.new_batch()
        self.env.reset()
        self.t = 0       
    

    def run(self, test_mode=False):
        """运行一个完整的episode
        
        Args:
            test_mode: 是否为测试模式。测试模式下不会更新训练统计信息。
            
        Returns:
            EpisodeBatch: 包含当前episode的所有转移数据的batch
        """
        # 重置环境和batch
        self.reset()

        terminated = False  # episode是否结束
        episode_return = 0  # episode累积奖励
        
        # 如果是RNN智能体,初始化隐藏状态
        if self.args.agent == "rnn":
            self.mac.init_hidden(batch_size=self.batch_size)

        # 运行episode直到结束
        while not terminated:#类似于DTQN中的step()采集信息,step()通过agent.observe（）将信息保存在context和replay_buffer
            # 收集当前时间步的环境信息
            pre_transition_data = {
                "state": [self.env.get_state()],  # 环境状态
                "avail_actions": [self.env.get_avail_actions()],  # 可用动作
                "obs": [self.env.get_obs()]  # 观察
            }
            # 更新batch中的数据
            # 通过分析transmix项目的代码,可以看到batch的内容:
            # 1. 在episode_buffer.py中的EpisodeBatch类初始化时:
            #    - self.data.transition_data = {} 存储每个时间步的数据
            #    - self.data.episode_data = {} 存储整个episode的数据
            #
            # 2. 在_setup_data()方法中:
            #    - scheme定义了所有数据字段的形状和类型
            #    - transition_data包含:state,obs,avail_actions,actions,reward等
            #    - episode_data用于存储episode级别的常量数据
            #
            # 3. 在learner.py的train()方法中可以看到使用这些数据:
            #    rewards = batch["reward"][:,:-1] 
            #    actions = batch["actions"][:,:-1]
            #    terminated = batch["terminated"][:,:-1]
            #print("obs shape:", pre_transition_data["obs"][0].shape)
            
            self.batch.update(pre_transition_data, ts=self.t)
            #print("obs shape after batch update: {}".format(self.batch["obs"].shape))

            # 让智能体选择动作
            # 传入当前时间步之前的所有经验batch
            # 返回大小为1的batch中每个智能体的动作
            # 传入的batch包含以下数据:
            # 1. transition_data - 每个时间步的数据,形状为(batch_size, max_seq_length, *)
            #    - state: [batch_size, max_seq_length, state_dim] 环境状态
            #    - obs: [batch_size, max_seq_length, n_agents, obs_dim] 每个智能体的观察
            #    - avail_actions: [batch_size, max_seq_length, n_agents, n_actions] 每个智能体的可用动作
            #    - filled: [batch_size, max_seq_length, 1] 数据是否有效
            #
            # 2. episode_data - 整个episode共享的数据,形状为(batch_size, *)
            #    - 用于存储episode级别的常量数据
            #
            # 在这里:
            # - batch_size=1 表示单个episode
            # - t_ep=self.t 表示当前时间步
            # - t_env用于epsilon衰减
            # - test_mode决定是否使用探索
            # 当timestep=t_ep时,self.batch中包含:
            # 1. transition_data:
            #    - state: [1, t_ep+1, state_dim] 环境状态序列
            #    - obs: [1, t_ep+1, n_agents, obs_dim] 每个智能体的观察序列 
            #    - avail_actions: [1, t_ep+1, n_agents, n_actions] 每个智能体的可用动作序列
            #    - actions: [1, t_ep, n_agents, 1] 每个智能体的历史动作序列(不含当前步)
            #    - reward: [1, t_ep, 1] 历史奖励序列(不含当前步)
            #    - terminated: [1, t_ep, 1] 历史终止状态序列(不含当前步)
            #
            # 2. episode_data: 整个episode共享的数据
            #    - state_shape: [1, state_dim] 状态空间维度
            #    - obs_shape: [1, obs_dim] 观察空间维度
            #    - n_agents: [1, 1] 智能体数量
            #    - episode_length: [1, 1] 回合长度
            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
            
            #传入0-t_ep + 1的batch，使得action_selector能够基于历史信息做出决策
            
            # 在transmix中:
            # 1. 每个智能体都会通过mac.select_actions()选择动作
            # 2. 所有智能体的动作被收集到actions中,形状为[1, n_agents] 
            # 3. actions[0]取出这些动作,传给环境同时执行
            # 4. 环境会根据所有智能体的联合动作计算下一状态和奖励
            # 在MARL环境中,step()方法通常返回(reward, terminated, info)
            # 而不是像单智能体环境那样返回(next_obs, reward, terminated, info)
            # 这是因为在MARL中:
            # 1. 每个智能体都有自己独特的观察视角,观察空间可能不同:
            #    - 智能体A可能只能看到周围一定范围
            #    - 智能体B可能有不同的传感器输入
            #    - 所以需要通过env.get_obs()为每个智能体单独获取观察
            # 2. 存在两个层次的状态:
            #    - 局部观察:每个智能体通过get_obs_agent(agent_id)获取
            #    - 全局状态:通过env.get_state()获取完整环境信息
            # 3. 为了统一接口和提高灵活性:
            #    - step()只返回共享信息(reward、terminated等)
            #    - 观察和状态通过专门的接口获取
            #    - 这样可以支持不同的观察方式和状态表示
            # 4. 这种设计也便于:
            #    - 实现集中式训练分布式执行(CTDE)
            #    - 处理部分可观察性(POMDP)场景
            #    - 支持异构智能体系统
            reward, terminated, env_info = self.env.step(actions[0])  # actions[0]包含所有智能体动作
            episode_return += reward  # 累积奖励

            # 收集执行动作后的信息
            # post_transition_data记录的数据来源:
            # - actions: 来自mac.select_actions()返回的动作
            # - reward: 来自env.step()返回的reward
            # - terminated: 来自env.step()返回的terminated,但需要判断是否是因为时间限制而终止
            post_transition_data = {
                "actions": actions,  # 来自mac.select_actions()选择的动作
                "reward": [(reward,)],  # 来自env.step()返回的奖励
                "terminated": [(terminated != env_info.get("episode_limit", False),)],  # 来自env.step()返回的终止状态
            }
            # 更新batch
            self.batch.update(post_transition_data, ts=self.t)
            

            self.t += 1  # 更新时间步，t_ep

        # episode结束后收集最后一个状态的信息
        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()]
        }
        self.batch.update(last_data, ts=self.t)

        # 在最后一个状态选择动作(用于计算target Q值)
        actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        self.batch.update({"actions": actions}, ts=self.t)


        # 更新统计信息
        cur_stats = self.test_stats if test_mode else self.train_stats  # 选择测试或训练统计
        cur_returns = self.test_returns if test_mode else self.train_returns  # 选择测试或训练回报
        log_prefix = "test_" if test_mode else ""  # 日志前缀
        
        # 更新环境信息统计
        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)  # episode计数
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)  # episode长度

        # 训练模式下更新环境交互总步数
        if not test_mode:
            self.t_env += self.t

        cur_returns.append(episode_return)  # 记录episode回报

        # 记录日志:测试模式下达到指定episode数,或训练模式下达到记录间隔
        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            # 记录epsilon值(如果使用)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        return self.batch

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
        stats.clear()
