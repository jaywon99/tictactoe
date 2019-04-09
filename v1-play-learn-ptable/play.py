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

dual = agent.DualAgent(agent1, agent2)

env = gym.getEnv()

TO_MARKER={-1: 'O', 1: 'X', 0: '='}
count = {-1: 0, 1: 0, 0: 0}
for step in range(1):
    winner = utils.play(env, dual, render = True)
    print(winner)
    print("WINNER", TO_MARKER[winner])
    env.render()

    count[winner] += 1
    if winner == -1:
        sys.exit(0)

# p_table.save('p_table.dat')
print(count)
