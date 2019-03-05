import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "lib"))

import tictactoe.gym as gym
import tictactoe.agent as agent
from tictactoe.utils import compact_observation 

MARKER={-1:'O', 0:'=', 1:'X'}
def find_next(env, seq, state, reward, done):
    if done:
        print(seq, MARKER[reward], compact_observation(state))
        return

    actions = env.available_actions()

    for action in actions:
        memento = env.create_memento()
        (state, reward, done, _) = env.step(action)
        find_next(env, seq+str(action), state, reward, done)
        env.set_memento(memento)

env = gym.getEnv()
env.reset()
find_next(env, "", None, None, False)

