''' implement negamax algorithm '''

import random
import math

import tictactoe.gym as gym
import tictactoe.agent as agent
import tictactoe.utils as utils

from .transposition import ABPTranspositionTable

def negamax_alpha_beta_pruning(tp, env, state=None, reward=0, done=False, depth=10, alpha=-math.inf, beta=math.inf):
    ''' implement negamax algorithm with alpha-beta purning
    https://en.wikipedia.org/wiki/Negamax
    '''
    # negamax.counter += 1

    # CHECK LEAF NODE / DO NOT NEED TO CHECK DEPTH = 0 BECASE TicTacToe is too small
    if done:
        if reward == 0:
            return (0, None)
        # return len(t.seq) - 10 # how to get score??
        return (-depth, None) # how to get score??

    orig_alpha = alpha

    # Transposition Table related work
    _id = utils.board_to_id(state)
    cache = tp.get(_id)
    if cache and cache['depth'] >= depth:
        if cache['flag'] == tp.EXACT:
            return cache['value']
        elif cache['flag'] == tp.LOWERBOUND:
            alpha = max(alpha, val)
        elif cache['flag'] == tp.UPPERBOUND:
            beta = min(beta, val)
        if alpha >= beta:
            return cache['value']
    # else:
    #     print("MISS", t.seq)

    # RECURSIVE
    actions = env.available_actions()
    random.shuffle(actions)
    best_score = -math.inf
    best_actions = []
    for action in actions:
        memento = env.create_memento()
        (state, reward, done, _) = env.step(action)
        score, _ = -negamax(env, state, reward, done, depth-1, alpha=-beta, beta=-alpha)
        score = -score # negamax
        env.set_memento(memento)

        # just pick up 1 first best move (random.shuffle make randomness)
        if score > best_score:
            best_score = score
            best_actions = [action]
        elif score == best_score:
            best_actions.append(action)

        if alpha < score:
            alpha = score
            # 결국 alpha = max(alpha, best_score)
            if alpha > beta:
                break

    if best_score <= orig_alpha:
        flag = tp.UPPERBOUND
    elif best_score >= beta:
        flag = tp.LOWERBOUND
    else:
        flag = tp.EXACT

    tp.put(key=_id, depth=depth, value=best_score, flag=flag)

    return random.choice(best_scores)


class ABPAgent(agent.AbstractAgent):
    def __init__(self, env):
        super().__init__()
        self.env = env
        self.tp = ABPTranspositionTable()

    def _next_action(self, state, availables_actions):
        negamax.counter = 0
        (_, next_action) = negamax_alpha_beta_pruning(self.tp, self.env, state, alpha=-math.inf, beta=math.inf)
        return next_action

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

