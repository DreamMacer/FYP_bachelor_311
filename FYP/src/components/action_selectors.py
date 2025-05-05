import torch as th
from torch.distributions import Categorical
from .epsilon_schedules import DecayThenFlatSchedule
REGISTRY = {}


class MultinomialActionSelector():

    def __init__(self, args):
        self.args = args

        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
                                              decay="linear")
        self.epsilon = self.schedule.eval(0)
        self.test_greedy = getattr(args, "test_greedy", True)

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):
        masked_policies = agent_inputs.clone()
        masked_policies[avail_actions == 0.0] = 0.0

        self.epsilon = self.schedule.eval(t_env)

        if test_mode and self.test_greedy:
            picked_actions = masked_policies.max(dim=2)[1]
        else:
            picked_actions = Categorical(masked_policies).sample().long()

        return picked_actions


REGISTRY["multinomial"] = MultinomialActionSelector


class EpsilonGreedyActionSelector():
    """
    ε-贪心动作选择器,用于在训练和测试时选择动作
    
    主要功能:
    1. 根据ε概率在随机动作和最优动作之间选择
    2. ε值随训练时间衰减,逐渐从探索转向利用
    3. 测试时使用纯贪心策略(ε=0)
    """

    def __init__(self, args):
        self.args = args
        # 创建ε衰减调度器,控制探索率的变化
        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
                                              decay="linear")
        self.epsilon = self.schedule.eval(0)  # 初始化ε值

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):
        """
        选择动作的主要逻辑
        agent_inputs: Q值张量 [batch_size, n_agents, n_actions] 
        avail_actions: 可用动作掩码 [batch_size, n_agents, n_actions]
        t_env: 当前环境步数,用于更新ε
        test_mode: 是否处于测试模式
        """
        # 根据当前步数更新ε值
        self.epsilon = self.schedule.eval(t_env)

        if test_mode:
            # 测试模式下使用纯贪心策略
            self.epsilon = 0.0

        # 将不可用动作的Q值设为负无穷,确保不会被选中
        masked_q_values = agent_inputs.clone()
        masked_q_values[avail_actions == 0.0] = -float("inf")

        # 生成随机数决定是否进行随机探索
        random_numbers = th.rand_like(agent_inputs[:, :, 0])
        pick_random = (random_numbers < self.epsilon).long()
        
        # 从可用动作中随机采样
        random_actions = Categorical(avail_actions.float()).sample().long()

        # 根据pick_random选择随机动作或最优动作
        picked_actions = pick_random * random_actions + (1 - pick_random) * masked_q_values.max(dim=2)[1]
        return picked_actions


REGISTRY["epsilon_greedy"] = EpsilonGreedyActionSelector
