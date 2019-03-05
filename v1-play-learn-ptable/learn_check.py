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
print("TOTAL", p_table.step, "ROUND PLAYED")

player1 = SmartAgent(p_table, debug=False)
player2 = SmartAgent(p_table, debug=False)

env = gym.getEnv()
dual = agent.DualAgent(player1, player2)

dual_x = agent.DualAgent(player1, agent.RandomAgent())
dual_o = agent.DualAgent(agent.RandomAgent(), player2)

MAX=10000
STEP=1000
# MAX=1000
# STEP=100

for step in range(p_table.step, p_table.step+MAX, STEP):
    for step1 in range(STEP):
        winner = utils.play(env, dual)

        if winner == 0: # TIE
            player1.feedback(0.0)
            player2.feedback(0.0)
        elif winner == 1: # X win
            player1.feedback(1.0)
            player2.feedback(-1.0)
        else: # O win
            player1.feedback(-1.0)
            player2.feedback(1.0)
        p_table.next_step()

    p_table.save('p_table.dat')

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


