''' ptable learning module '''

import tictactoe.utils as utils
from tictactoe.agent import DualAgent

from .agent import MyAgent

AGENT1_DATA_PATH = "./models/ptable1.dat"
AGENT2_DATA_PATH = "./models/ptable2.dat"

def build_agent():
    ''' build agent for learning '''
    # create ptable agent1 & agent2
    agent1 = MyAgent(learning_rate=0.1, debug=False)
    agent2 = MyAgent(learning_rate=0.1, debug=False)

    agent1.load(AGENT1_DATA_PATH)
    agent2.load(AGENT2_DATA_PATH)

    return agent1, agent2

MAX = 1000000
STEP = 1000

def learn(env, best_agent):
    ''' learn using ptable '''

    agent1, agent2 = build_agent()

    dual = DualAgent(agent1, agent2)
    dual_x = DualAgent(agent1, best_agent)
    dual_o = DualAgent(best_agent, agent2)

    full_tie = 0

    for step in range(0, MAX, STEP):
        dual.set_train_mode(True)
        for _ in range(STEP):
            utils.play(env, dual, render=False)

        agent1.save(AGENT1_DATA_PATH)
        agent2.save(AGENT2_DATA_PATH)

        dual.set_train_mode(False)

        count1 = {-1: 0, 1: 0, 0: 0}
        for _ in range(100):
            winner = utils.play(env, dual_o)
            count1[winner] += 1

        count2 = {-1: 0, 1: 0, 0: 0}
        for _ in range(100):
            winner = utils.play(env, dual_x)
            count2[winner] += 1

        print(step+STEP, count1[1], count1[-1], count1[0], count2[1], count2[-1], count2[0])

        # 5 consecutive full tie, learning completed
        if count1[-1] == 0 and count2[1] == 0:
            full_tie += 1
            if full_tie == 5:
                break
        else:
            full_tie = 0
