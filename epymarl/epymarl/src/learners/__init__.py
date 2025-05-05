from .q_learner import QLearner
from .coma_learner import COMALearner
from .qtran_learner import QLearner as QTranLearner
from .actor_critic_learner import ActorCriticLearner
from .actor_critic_pac_learner import PACActorCriticLearner
from .actor_critic_pac_dcg_learner import PACDCGLearner
from .maddpg_learner import MADDPGLearner
from .ppo_learner import PPOLearner
from .Qlearner import QLearner as DTQNLearner #agent为rnn,mixingnetwork为 transmix/transformer
from .dtqnlearner import QLearner as dtqnlearner #agent 为dtqn,mixingnetwork为 transmix

REGISTRY = {}
REGISTRY["q_learner"] = QLearner
REGISTRY["coma_learner"] = COMALearner
REGISTRY["qtran_learner"] = QTranLearner
REGISTRY["actor_critic_learner"] = ActorCriticLearner
REGISTRY["maddpg_learner"] = MADDPGLearner
REGISTRY["ppo_learner"] = PPOLearner
REGISTRY["pac_learner"] = PACActorCriticLearner
REGISTRY["pac_dcg_learner"] = PACDCGLearner
REGISTRY["dtqn_learner"] = DTQNLearner
REGISTRY["dtqnlearner"] = dtqnlearner
