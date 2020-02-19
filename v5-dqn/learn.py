import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "lib"))

import tictactoe.gym as gym
import tictactoe.agent as agent
import tictactoe.utils as utils

from tictactoe_dqn_cp import TicTacToeDQN
from dqnagent import DQNAgent

# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

dqn1 = TicTacToeDQN()
dqn2 = TicTacToeDQN()
sess = tf.Session()
sess.run(tf.global_variables_initializer())
dqn1.set_session(sess)
dqn2.set_session(sess)

player1 = DQNAgent(1, dqn1)
player2 = DQNAgent(-1, dqn2)
# player2 = agent.RandomAgent()

env = gym.getEnv()
dual = agent.DualAgent(player1, player2)

EPOCH = 1000000
STEP = 1000
for x in range(1, EPOCH, STEP):
    dual.set_train_mode(True)
    for step1 in range(STEP):
        reward = utils.play(env, dual)
        
    print("STEP", x+step1)
    dqn1.save("./models/dqn1.ckpt")
    dqn2.save("./models/dqn2.ckpt")

