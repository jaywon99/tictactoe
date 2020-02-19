import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "lib"))

import tictactoe.gym as gym
import tictactoe.agent as agent
import tictactoe.utils as utils

from agent import MyAgent
from tictactoe.negamax.negamax import NegamaxAgent

env = gym.getEnv()
nmplayer = NegamaxAgent(env)

agent1 = MyAgent(learning_rate=0.1, discount_rate=0.9, exploit_rate=0.2)
agent1.load("./models/q1.dat")
agent2 = MyAgent(learning_rate=0.1, discount_rate=0.9, exploit_rate=0.2)
agent2.load("./models/q2.dat")

dual_x = agent.DualAgent(agent1, nmplayer)
dual_o = agent.DualAgent(nmplayer, agent2)
dual   = agent.DualAgent(agent1, agent2)

dual.set_train_mode(False)
for step1 in range(10000):
    winner, history = utils.play(env, dual_o, recorded = True)
    if winner != 0:
        print("DUAL_O", winner, "".join([str(h) for h in history]))
        break

for step1 in range(10000):
    winner = utils.play(env, dual_x)
    if winner != 0:
        print("DUAL_X", winner, "".join([str(h) for h in history]))
        break

for step1 in range(10000):
    winner = utils.play(env, dual)
    if winner != 0:
        print("DUAL", winner, "".join([str(h) for h in history]))
        break

