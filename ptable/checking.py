''' Checking ptable agent working well '''

import tictactoe.agent as agent
import tictactoe.utils as utils

from .agent import MyAgent

AGENT1_DATA_PATH = "./models/ptable1.dat"
AGENT2_DATA_PATH = "./models/ptable2.dat"

def build_agent():
    ''' build agent for learning '''
    # create ptable agent1 & agent2
    agent1 = MyAgent(learning_rate=0.5, random_rate=0.0, debug=False)
    agent2 = MyAgent(learning_rate=0.5, random_rate=0.0, debug=False)

    agent1.load(AGENT1_DATA_PATH)
    agent2.load(AGENT2_DATA_PATH)

    return agent1, agent2

def checking(env, best_player):
    ''' checking ptable agent working well '''

    agent1, agent2 = build_agent()

    dual_x = agent.DualAgent(agent1, best_player)
    dual_o = agent.DualAgent(best_player, agent2)
    dual = agent.DualAgent(agent1, agent2)
    dual.set_train_mode(False)

    for _ in range(10000):
        winner, history = utils.play(env, dual_x, recorded=True)
        if winner != 0:
            print("DUAL_X", winner, "".join([str(h) for h in history]))
            break

    for _ in range(10000):
        winner, history = utils.play(env, dual_o, recorded=True)
        if winner != 0:
            print("DUAL_O", winner, "".join([str(h) for h in history]))
            break

    for _ in range(10000):
        winner, history = utils.play(env, dual, recorded=True)
        if winner != 0:
            print("DUAL", winner, "".join([str(h) for h in history]))
            break
