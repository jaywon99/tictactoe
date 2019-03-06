import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "lib"))

import tictactoe.gym as gym
import tictactoe.agent as agent
import tictactoe.utils as utils

from ptable_agent import SmartAgent
from ptable import PredictionTable

p_table = PredictionTable(learning_rate=0.5)
p_table.load('p_table.dat')

agent1 = SmartAgent(p_table, debug=False)
agent2 = SmartAgent(p_table, debug=False)

env = gym.getEnv()

print(utils.test_player1(env, agent1))
print(utils.test_player2(env, agent2))
