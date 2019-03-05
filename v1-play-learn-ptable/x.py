import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "lib"))

import tictactoe.gym as gym
import tictactoe.utils as utils

from ptable import PredictionTable

p_table = PredictionTable(learning_rate=0.5)
p_table.load('p_table.dat')

seq = sys.argv[1]

env = gym.getEnv()
next_state = env.reset()
for action in seq:
    (next_state, reward, done, _) = env.step(int(action))
_id = utils.compact_observation(next_state)

p = None if _id not in p_table.pred_table else p_table.pred_table[_id]
print(seq, _id, p, p_table.lookup(_id))
