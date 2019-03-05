import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "lib"))

import tictactoe.gym as gym
import tictactoe.agent as agent
import tictactoe.utils as utils
from ptable_agent import SmartAgent

from ptable import PredictionTable

p_table = PredictionTable(learning_rate=0.5)
p_table.load('tictactoe.dat')

player1 = SmartAgent(p_table, debug=False)
player2 = SmartAgent(p_table, debug=False)

env = gym.getEnv()
dual = agent.DualAgent(player1, player2)

dual_x = agent.DualAgent(player1, agent.RandomAgent())
dual_o = agent.DualAgent(agent.RandomAgent(), player2)

seq = sys.argv[1]

seq1 = [int(s) for i, s in enumerate(seq) if i%2==0]
seq2 = [int(s) for i, s in enumerate(seq) if i%2==1]

TO_MARKER={-1: 'O', 1: 'X', 0: '='}
# print(line.strip(), line.strip().split(' '))
player1.set_queue(seq1)
player2.set_queue(seq2)
winner = utils.play(env, dual)
print(winner)
print("WINNER", TO_MARKER[winner])
env.render()

if winner == 0: # TIE
    player1.feedback(0.0)
    player2.feedback(0.0)
elif winner == 1: # X win
    player1.feedback(1.0)
    player2.feedback(-1.0)
else: # O win
    player1.feedback(-1.0)
    player2.feedback(1.0)
print(p_table.pred_table)



