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

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):#输入参数是agent_outputs，动作的概率分布
        masked_policies = agent_inputs.clone()
        masked_policies[avail_actions == 0.0] = 0.0#复制策略输出，并将不可用动作的概率设置为 0

        self.epsilon = self.schedule.eval(t_env)

        if test_mode and self.test_greedy:
            picked_actions = masked_policies.max(dim=2)[1]
        else:
            picked_actions = Categorical(masked_policies).sample().long()

        return picked_actions


REGISTRY["multinomial"] = MultinomialActionSelector


class EpsilonGreedyActionSelector():

    def __init__(self, args):
        self.args = args

        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
                                              decay="linear")
        self.epsilon = self.schedule.eval(0)

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):

        # Assuming agent_inputs is a batch of Q-Values for each agent bav
        self.epsilon = self.schedule.eval(t_env)

        if test_mode:
            # Greedy action selection only
            self.epsilon = 0.0

        # mask actions that are excluded from selection
        masked_q_values = agent_inputs.clone()
        masked_q_values[avail_actions == 0.0] = -float("inf")  # should never be selected!

        random_numbers = th.rand_like(agent_inputs[:, :, 0])#生成与每个智能体对应的随机数
        pick_random = (random_numbers < self.epsilon).long()#将每个随机数与当前 ε 值比较。如果随机数小于 ε，则表示随机选择动作，否则执行贪心策略。一个形状为 [batch_size, n_agents] 的张量，值为 0 或 1
        random_actions = Categorical(avail_actions.float()).sample().long()#形状为 [batch_size, n_agents] 的二维张量，每个值表示随机选择的动作索引

        picked_actions = pick_random * random_actions + (1 - pick_random) * masked_q_values.max(dim=2)[1]
        return picked_actions


REGISTRY["epsilon_greedy"] = EpsilonGreedyActionSelector
