''' implement negamax algorithm '''

import random

import tictactoe.agent as agent
import tictactoe.utils as utils

from .transposition import TranspositionTable

tp = TranspositionTable()

def negamax(env, state=None, reward=0, done=False, depth=10):
    ''' implement negamax algorithm
    https://en.wikipedia.org/wiki/Negamax
    '''
    # negamax.counter += 1

    # Transposition Table related work
    _id = utils.board_to_id(state)
    cache = tp.get(_id)
    if cache is not None: # BUG FIX! cache can be 0, so should check None
        # case 1
        # return cache
        # case 2
        return random.choice(cache)

    # CHECK LEAF NODE / DO NOT NEED TO CHECK DEPTH = 0 BECASE TicTacToe is too small
    if done:
        if reward == 0:
            return (0, None)
        # return len(t.seq) - 10 # how to get score??
        return (-depth, None) # how to get score??

    # RECURSIVE
    actions = env.available_actions()
    best_score = -11
    best_scores = []
    for action in actions:
        memento = env.create_memento()
        (state, reward, done, _) = env.step(action)
        score, _ = negamax(env, state, reward, done, depth-1)
        score = -score # negamax
        env.set_memento(memento)

        if score > best_score:
            best_score = score
            best_scores = [(score, action)]
        elif score == best_score:
            best_scores.append((score, action))

    # case 1: choose random value 1 time
    # choosed_result = random.choice(best_scores)
    # tp.put(_id, choosed_result)
    # return choosed_result

    # case 2: choose random value every time
    tp.put(_id, best_scores)
    return random.choice(best_scores)

class NegamaxAgent(agent.AbstractAgent):
    ''' negamax tic-tac-toe agent '''
    def __init__(self, env):
        super().__init__()
        self.env = env

    def _next_action(self, state, available_actions):
        # return smart_turn(self.env)
        (_, next_action) = negamax(self.env, state)
        return next_action
