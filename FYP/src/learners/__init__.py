from .q_learner import QLearner
# from .coma_learner import COMALearner
# from .qtran_learner import QLearner as QTranLearner
from .q_learner0 import QLearnerorigin

REGISTRY = {}

REGISTRY["q_learner"] = QLearner
# REGISTRY["coma_learner"] = COMALearner
# REGISTRY["qtran_learner"] = QTranLearner
REGISTRY["q_learnerorigin"] = QLearnerorigin
