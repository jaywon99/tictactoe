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

MAX=1000000
STEP=100

full_tie = 0

for step in range(1, MAX, STEP):
    dual.set_train_mode(True)
    for step1 in range(STEP):
        winner = utils.play(env, dual, render=False)

    player1.save("./models/p1.dat")
    player2.save("./models/p2.dat")

    dual.set_train_mode(False)
    count1 = {-1: 0, 1: 0, 0: 0}
    for step1 in range(100):
        winner = utils.play(env, dual_o)
        count1[winner] += 1

    count2 = {-1: 0, 1: 0, 0: 0}
    for step1 in range(100):
        winner = utils.play(env, dual_x)
        count2[winner] += 1
    print(step+STEP-1, count1[1], count1[-1], count1[0], count2[1], count2[-1], count2[0])

    # 5 consecutive full tie, learning completed
    if count1[0] == 100 and count2[0] == 100:
        full_tie += 1
        if full_tie == 5:
            break
    else:
        full_tie = 0

