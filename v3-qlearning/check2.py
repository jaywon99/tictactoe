import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "lib"))

import tictactoe.gym as gym
import tictactoe.agent as agent
import tictactoe.utils as utils

from qagent import QAgent
from qlearning import qlearning

env = gym.getEnv()

q1 = qlearning(value=0, n_actions=9, learning_rate=0.1, discount=0.9, exploit_rate=0.1)
q1.load("qq.dat")

# agent1 = QAgent(q1)
agent1 = agent.RandomAgent()
agent2 = QAgent(q1)

dual = agent.DualAgent(agent1, agent2)

def test(env, dual, debug=False):
    # agent1.set_mode(0)
    agent2.set_mode(0)
    return utils.play(env, dual, debug)

scores = {-1: 0, 0: 0, 1: 0}
for i in range(100):
    scores[test(env, dual, True)] += 1
    print()
print("SCORE", scores)

