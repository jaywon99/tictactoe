import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "lib"))

import tictactoe.gym as gym
import tictactoe.agent as agent
import tictactoe.utils as utils

from qagent import QAgent
from qlearning import qlearning

q1 = qlearning(value=0, n_actions=9, learning_rate=0.1, discount=0.9, exploit_rate=0.2)
q2 = qlearning(value=0, n_actions=9, learning_rate=0.1, discount=0.9, exploit_rate=0.2)
q1.load("q1.dat")
q2.load("q2.dat")

agent1 = QAgent(q1)
agent2 = QAgent(q2)

env = gym.getEnv()

print(utils.test_player1(env, agent1))
print(utils.test_player2(env, agent2))
