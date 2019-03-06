import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "lib"))

import tictactoe.gym as gym
import tictactoe.agent as agent
import tictactoe.utils as utils

from ptable_agent import SmartAgent
from ptable import PredictionTable

p_table1 = PredictionTable(learning_rate=0.5)
p_table2 = PredictionTable(learning_rate=0.5)

player1 = SmartAgent(p_table1, debug=False)
player2 = SmartAgent(p_table2, debug=False)

env = gym.getEnv()
dual = agent.DualAgent(player1, player2)

STEP = 10000
for x in range(10000):
    for step1 in range(100):
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
    p_table1.next_step()

player3 = SmartAgent(p_table1, random_rate=0.0, debug=True)
player4 = SmartAgent(p_table2, random_rate=0.0, debug=True)
dual = agent.DualAgent(player3, player4)

TO_MARKER={-1: 'O', 1: 'X', 0: '='}
count = {-1: 0, 1: 0, 0: 0}
for step in range(1):
    winner = utils.play(env, dual)
    print("WINNER", TO_MARKER[winner])
    env.render()

    count[winner] += 1
    if winner == -1:
        sys.exit(0)

# p_table.save('p_table.dat')
print(count)
