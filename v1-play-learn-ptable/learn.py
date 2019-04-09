import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "lib"))

import tictactoe.gym as gym
import tictactoe.agent as agent
import tictactoe.utils as utils

from agent import SmartAgent

player1 = SmartAgent(learning_rate=0.5, debug=False)
player1.load("./models/p1.dat")
player2 = SmartAgent(learning_rate=0.5, debug=False)
player2.load("./models/p2.dat")

env = gym.getEnv()
dual = agent.DualAgent(player1, player2)

dual_x = agent.DualAgent(player1, agent.RandomAgent())
dual_o = agent.DualAgent(agent.RandomAgent(), player2)

MAX=10000
STEP=1000
# MAX=1000
# STEP=100

for step in range(1, MAX, STEP):
    for step1 in range(STEP):
        winner = utils.play(env, dual, step+step1, feedback=True, render=False)

    player1.save("p1.dat")
    player2.save("p2.dat")

    count = {-1: 0, 1: 0, 0: 0}
    for step1 in range(1000):
        winner = utils.play(env, dual_o)
        count[winner] += 1
    print("STEP", step, "SMART_O", count)

    count = {-1: 0, 1: 0, 0: 0}
    for step1 in range(1000):
        winner = utils.play(env, dual_x)
        count[winner] += 1
    print("STEP", step, "SMART_X", count)

    winner = utils.play(env, dual)
    print("STEP", step, "WINNER", winner)
    if winner != 0:
        env.render()


