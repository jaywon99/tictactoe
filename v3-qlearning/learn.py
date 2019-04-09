import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "lib"))

import tictactoe.gym as gym
import tictactoe.agent as agent
import tictactoe.utils as utils

from agent import MyAgent

env = gym.getEnv()

agent1 = MyAgent(learning_rate=0.1, discount_rate=0.9, exploit_rate=0.2)
agent1.load("./models/q1.dat")
agent2 = MyAgent(learning_rate=0.1, discount_rate=0.9, exploit_rate=0.2)
agent2.load("./models/q2.dat")

dual = agent.DualAgent(agent1, agent2)

agent4 = agent.RandomAgent()
dual2 = agent.DualAgent(agent1, agent4)

def repeat(env, dual, dual2, n):
    wins = 0
    for i in range(1, n+1):
        utils.play(env, dual, feedback = True)

        if i % 1000 == 0:
            print("EPOCH", i, end=' ')
            cnt = {-1: 0, 0: 0, 1: 0}
            for j in range(100):
                cnt[utils.play(env, dual2, feedback = False)] += 1
            print("PLAY 100, SCORE", cnt)
            agent1.save("./models/q1-"+str(i)+".dat")
            agent2.save("./models/q2-"+str(i)+".dat")
    agent1.save("./models/q1.dat")
    agent2.save("./models/q2.dat")
    return wins

print("WINS", repeat(env, dual, dual2, 10000000))

scores = {-1: 0, 0: 0, 1: 0}
for i in range(100):
    scores[utils.play(env, dual2, False)] += 1
    # print()
print("SCORE", scores)

