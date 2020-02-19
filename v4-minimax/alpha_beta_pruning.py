import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "lib"))

import random

import tictactoe.gym as gym
import tictactoe.agent as agent
import tictactoe.utils as utils

from transposition import ABPTranspositionTable

tp = ABPTranspositionTable()
INF = float('infinity')
LOWERBOUND, EXACT, UPPERBOUND = -1,0,1

def negamax(env, state, reward, done, depth, alpha, beta):
    negamax.counter += 1
    best_score = -11
    alphaOrig = alpha

    # Transposition Table related work
    _id = utils.board_to_id(state)
    cache = tp.get(_id)
    if cache:
        val = cache['value']
        if cache['flag'] == EXACT:
            return val
        elif cache['flag'] == LOWERBOUND:
            alpha = max(alpha, val)
        elif cache['flag'] == UPPERBOUND:
            beta = min(beta, val)
        if alpha >= beta:
            return val
    # else:
    #     print("MISS", t.seq)

    # CHECK LEAF NODE / DO NOT NEED TO CHECK DEPTH = 0 BECASE TicTacToe is too small
    if done and reward == 0:
        return 0
    if done:
        # return len(t.seq) - 10 # how to get score??
        return -depth # how to get score??

    # RECURSIVE
    actions = env.available_actions()
    for action in actions:
        memento = env.create_memento()
        (state, reward, done, _) = env.step(action)
        score = -negamax(env, state, reward, done, depth-1, -beta, -alpha)
        env.set_memento(memento)

        alpha = max(alpha, score)
        if alpha >= beta:
            tp.put(key=_id, depth=depth, value=score, flag=LOWERBOUND)
            return score

#    ttable[_id] = best_score
    tp.put(key=_id, depth=depth, value=alpha, 
           flag=UPPERBOUND if score <= alphaOrig else EXACT)

    return alpha

    # return best_score

def smart_turn(env):
    scores = {}
    actions = env.available_actions()

    for action in actions:
        memento = env.create_memento()
        (state, reward, done, _) = env.step(action)
        score = -negamax(env, state, reward, done, 10, -INF, INF)
        env.set_memento(memento)

        scores[action] = score

    max_scores = max(scores.values())
    highest = [k for k,v in scores.items() if v == max_scores]
    print(scores, max_scores, max(scores.values()), highest)
    pos = random.choice(highest)

    return pos

class ABPAgent(agent.AbstractAgent):
    def __init__(self, env):
        super().__init__()
        self.env = env

    def _next_action(self, obj, availables_actions):
        negamax.counter = 0
        return smart_turn(self.env)

if __name__ == "__main__":
    env = gym.getEnv()
    env.reset()

    done = False
    for c in sys.argv[1]:
        (state, reward, done, _) = env.step(int(c))
    env.render()

    while not done:
        negamax.counter = 0
        action = smart_turn(env)
        # print(negamax.counter)
        (state, reward, done, _) = env.step(action)
        env.render()

