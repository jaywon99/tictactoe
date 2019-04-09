import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "lib"))

import tictactoe.gym as gym
import tictactoe.agent as agent
import tictactoe.utils as utils

from agent import SmartAgent

agent1 = SmartAgent(learning_rate=0.5, random_rate=0.0, debug=False)
agent1.load("./models/p1.dat")
agent2 = SmartAgent(learning_rate=0.5, random_rate=0.0, debug=False)
agent2.load("./models/p2.dat")

env = gym.getEnv()

print(utils.test_player1(env, agent1))
print(utils.test_player2(env, agent2))
