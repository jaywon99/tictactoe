import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "lib"))

import tictactoe.gym as gym
import tictactoe.agent as agent
import tictactoe.utils as utils

from qagent import QAgent
from qlearning import qlearning

env = gym.getEnv()

q1 = qlearning(value=0, n_actions=9, learning_rate=0.1, discount=0.9, exploit_rate=0.2)
q2 = qlearning(value=0, n_actions=9, learning_rate=0.1, discount=0.9, exploit_rate=0.2)
q1.load("q1.dat")
q2.load("q2.dat")

agent1 = QAgent(q1)
agent2 = QAgent(q2)

dual = agent.DualAgent(agent1, agent2)

agent4 = agent.RandomAgent()
dual2 = agent.DualAgent(agent1, agent4)

def learn(env, dual):
    agent1.set_mode(1)
    agent2.set_mode(1)
    next_state = env.reset()
    dual.reset()
    done = False
    action = None
    while not done:
        agent = dual.next_agent()
        if action != None:
            agent.feedback(next_state, reward)
        action = agent.next_action(next_state, env.available_actions())
        (next_state, reward, done, _) = env.step(action)

    # print()
    if reward == 1: # X win
        dual.agent1.feedback(next_state, 1)
        dual.agent2.feedback(next_state, -1)
    elif reward == -1: # O win
        dual.agent1.feedback(next_state, -1)
        dual.agent2.feedback(next_state, 1)
    else:
        dual.agent1.feedback(next_state, 0)
        dual.agent2.feedback(next_state, 0)
    # print()

def test(env, dual, debug=False):
    agent1.set_mode(0)
    agent2.set_mode(0)
    return utils.play(env, dual, debug)

def repeat(env, dual, dual2, n):
    wins = 0
    for i in range(1, n+1):
        learn(env, dual)

        if i % 1000 == 0:
            print("EPOCH", i, end=' ')
            cnt = {-1: 0, 0: 0, 1: 0}
            for j in range(100):
                cnt[test(env, dual2)] += 1
            print("PLAY 100, SCORE", cnt)
            q1.save("q1.dat")
            q2.save("q2.dat")
    q1.save("q1.dat")
    q2.save("q2.dat")
    return wins

print("WINS", repeat(env, dual, dual2, 10000000))

scores = {-1: 0, 0: 0, 1: 0}
for i in range(100):
    scores[test(env, dual2, False)] += 1
    # print()
print("SCORE", scores)

