import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "lib"))

import tictactoe.gym as gym
import tictactoe.agent as agent
import tictactoe.utils as utils

from agent import SmartAgent

from tictactoe.negamax.negamax import NegamaxAgent

env = gym.getEnv()

nmplayer = NegamaxAgent(env)

player1 = SmartAgent(learning_rate=0.5, debug=False)
player1.load("./models/p1.dat")
player2 = SmartAgent(learning_rate=0.5, debug=False)
player2.load("./models/p2.dat")
dual = agent.DualAgent(player1, player2)

dual_x = agent.DualAgent(player1, nmplayer)
dual_o = agent.DualAgent(nmplayer, player2)

MAX=10000
STEP=100

for step in range(1, MAX, STEP):
    for step1 in range(STEP):
        winner = utils.play(env, dual, step+step1, feedback=True, render=False)

    player1.save("./models/p1.dat")
    player2.save("./models/p2.dat")

    count = {-1: 0, 1: 0, 0: 0}
    for step1 in range(100):
        winner = utils.play(env, dual_o)
        count[winner] += 1
    print("STEP", step+STEP-1, "X(NM)", count[1], "O(P)", count[-1], "TIE", count[0])

    count = {-1: 0, 1: 0, 0: 0}
    for step1 in range(100):
        winner = utils.play(env, dual_x)
        count[winner] += 1
    print("STEP", step+STEP-1, "X(P)", count[1], "O(NM)", count[-1], "TIE", count[0])


