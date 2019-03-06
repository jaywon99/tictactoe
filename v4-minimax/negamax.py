import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "lib"))

import random

import tictactoe.gym as gym
import tictactoe.agent as agent
import tictactoe.utils as utils

from transposition import TranspositionTable

tp = TranspositionTable()

def negamax(env, state, reward, done, depth):
    # negamax.counter += 1
    best_score = -11

    # Transposition Table related work
    _id = utils.compact_observation(state)
    cache = tp.get(_id)
    if cache:
        return cache

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
        score = -negamax(env, state, reward, done, depth-1)
        env.set_memento(memento)

        if score > best_score:
            best_score = score

    tp.put(_id, best_score)

    return best_score

def smart_turn(env):
    scores = {}
    actions = env.available_actions()

    for action in actions:
        memento = env.create_memento()
        (state, reward, done, _) = env.step(action)
        score = -negamax(env, state, reward, done, 10)
        env.set_memento(memento)

        scores[action] = score

    max_scores = max(scores.values())
    highest = [k for k,v in scores.items() if v == max_scores]
    # print(scores, max_scores, max(scores.values()), highest)
    pos = random.choice(highest)

    return pos

class NegamaxAgent(agent.AbstractAgent):
    def __init__(self, env):
        super().__init__()
        self.env = env

    def _next_action(self, obj, availables_actions):
        return smart_turn(self.env)

if __name__ == "__main__":
    env = gym.getEnv()
    env.reset()

    done = False
    for c in sys.argv[1]:
        (state, reward, done, _) = env.step(int(c))
    env.render()

    while not done:
        # negamax.counter = 0
        action = smart_turn(env)
        # print(negamax.counter)
        (state, reward, done, _) = env.step(action)
        env.render()

