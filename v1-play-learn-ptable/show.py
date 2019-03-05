import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "lib"))

import tictactoe.gym as gym
import tictactoe.utils as utils

seq = sys.argv[1]

env = gym.getEnv()
next_state = env.reset()
for l in range(len(seq)):
    print("SEQ:", seq[:l+1])
    action = seq[l]
    (next_state, reward, done, _) = env.step(int(action))
    env.render()

    print("------------------")
    if done: break
