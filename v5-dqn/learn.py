import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "lib"))

import tictactoe.gym as gym
import tictactoe.agent as agent

from tictactoe_dqn_cp import TicTacToeDQN
from dqnagent import DQNAgent

import tensorflow as tf

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

def play(env, dual, render = False):
    player1.set_mode(1)
    player2.set_mode(1)
    next_state = env.reset()
    dual.reset()
    done = False
    action = None
    while not done:
        if render:
            env.render()
        agent = dual.next_agent()
        if action != None:
            agent.feedback(next_state, reward, 0)
        action = agent.next_action(next_state, env.available_actions())
        (next_state, reward, done, _) = env.step(action)
    if render:
        env.render()

    # print()
    if reward == 1: # X win
        dual.agent1.feedback(next_state, 1, 1)
        dual.agent2.feedback(next_state, -1, 1)
    elif reward == -1: # O win
        dual.agent1.feedback(next_state, -1, 1)
        dual.agent2.feedback(next_state, 1, 1)
    else:
        dual.agent1.feedback(next_state, 0, 1)
        dual.agent2.feedback(next_state, 0, 1)
    # print()

    return reward

EPOCH = 1000000
STEP = 1000
for x in range(1, EPOCH, STEP):
    for step1 in range(STEP):
        reward = play(env, dual)
        
    print("STEP", x+step1)
    dqn1.save("./models/dqn1.ckpt")
    dqn2.save("./models/dqn2.ckpt")

