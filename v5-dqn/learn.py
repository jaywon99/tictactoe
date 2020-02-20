import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "lib"))

import tictactoe.gym as gym
import tictactoe.agent as agent
import tictactoe.utils as utils

from agent import MyAgent

from tictactoe.negamax.negamax import NegamaxAgent

# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

env = gym.getEnv()

agent1 = MyAgent()
agent2 = MyAgent()

sess = tf.Session()
agent1.set_session(sess)
agent2.set_session(sess)
sess.run(tf.global_variables_initializer())

dual = agent.DualAgent(agent1, agent2)

nmplayer = NegamaxAgent(env)

dual_x = agent.DualAgent(agent1, nmplayer)
dual_o = agent.DualAgent(nmplayer, agent2)

EPOCH = 1000000
STEP = 1000
for step in range(1, EPOCH, STEP):
    dual.set_train_mode(True)
    for step1 in range(STEP):
        reward = utils.play(env, dual)

    print("STEP", step+step1)
    agent1.save("./models/dqn1.ckpt")
    agent1.save("./models/dqn2.ckpt")

    dual.set_train_mode(False)
    count1 = {-1: 0, 1: 0, 0: 0}
    for step1 in range(100):
        winner = utils.play(env, dual_o)
        count1[winner] += 1

    count2 = {-1: 0, 1: 0, 0: 0}
    for step1 in range(100):
        winner = utils.play(env, dual_x)
        count2[winner] += 1
    print(step+STEP, count1[1], count1[-1], count1[0], count2[1], count2[-1], count2[0])

    # 5 consecutive full tie, learning completed
    if count1[0] == 100 and count2[0] == 100:
        full_tie += 1
        if full_tie == 5:
            break
    else:
        full_tie = 0
