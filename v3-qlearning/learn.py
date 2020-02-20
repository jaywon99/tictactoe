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
dual = agent.DualAgent(agent1, agent2)

dual_x = agent.DualAgent(agent1, nmplayer)
dual_o = agent.DualAgent(nmplayer, agent2)

MAX=100000
STEP=100

full_tie = 0

for step in range(0, MAX, STEP):
    dual.set_train_mode(True)
    for step1 in range(STEP):
        utils.play(env, dual, render=False)

    agent1.save("./models/q1-"+str(step)+".dat")
    agent2.save("./models/q2-"+str(step)+".dat")

    dual.set_train_mode(False)
    count1 = {-1: 0, 1: 0, 0: 0}
    for step1 in range(100):
        winner = utils.play(env, dual_o)
        count1[winner] += 1

    count2 = {-1: 0, 1: 0, 0: 0}
    for step1 in range(100):
        winner = utils.play(env, dual_x)
        count2[winner] += 1
    print(step+STEP, count1[1], count1[-1], count1[0], count2[1], count2[-1], count2[0])

    # 5 consecutive full tie, learning completed
    if count1[0] == 100 and count2[0] == 100:
        full_tie += 1
        if full_tie == 5:
            break
    else:
        full_tie = 0

agent1.save("./models/q1.dat")
agent2.save("./models/q2.dat")

