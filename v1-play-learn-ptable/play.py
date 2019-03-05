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

agent1 = SmartAgent(p_table, debug=True)
agent2 = SmartAgent(p_table, debug=True)

dual = agent.DualAgent(agent1, agent2)

env = gym.getEnv()

TO_MARKER={-1: 'O', 1: 'X', 0: '='}
count = {-1: 0, 1: 0, 0: 0}
for step in range(1):
    winner = utils.play(env, dual)
    print(winner)
    print("WINNER", TO_MARKER[winner])
    env.render()

    count[winner] += 1
    if winner == -1:
        sys.exit(0)

# p_table.save('p_table.dat')
print(count)
