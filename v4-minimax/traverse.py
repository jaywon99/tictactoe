import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "lib"))

import tictactoe.gym as gym
import tictactoe.agent as agent
import tictactoe.test_utils as utils

import negamax
import alpha_beta_pruning

env = gym.getEnv()
agent = negamax.NegamaxAgent(env)

print(utils.test_player1(env, agent))
print(utils.test_player2(env, agent))

agent = alpha_beta_pruning.ABPAgent(env)

print(utils.test_player1(env, agent))
print(utils.test_player2(env, agent))

