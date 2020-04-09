''' implement negamax algorithm '''

import random
import math

from boardAI import AbstractPlayer, GameResult
from tictactoe import OptimalBoard, TicTacToeBoard
from tictactoe.selfplay import SelfPlayTicTacToeBoard as SP

from .transposition import TranspositionTable

class NegamaxPlayer(AbstractPlayer):
    ''' negamax tic-tac-toe agent '''
    DEPTH = 10

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tp = TranspositionTable()

    def serialize(self):
        return self.tp.serialize()

    def deserialize(self, obj):
        if obj != None:
            self.tp.deserialize(obj)

    def _choose(self, state, available_actions):
        # return smart_turn(self.env)
        # state는 0,1,-1로 이루어진 9칸 array
        # 1. board를 만들 필요가 있을까? 아니면 그냥 state로 동작할까?
        # state, color, available_actions로 negamax를 동작
        # available_actions가 negamax에 필요한가? 그렇지 않음
        my_color = TicTacToeBoard.COLOR_TO_INTERNAL[self.color]
        (_, next_action) = self.negamax(state, my_color, depth=NegamaxPlayer.DEPTH)
        return next_action

    def get_tp(self):
        return self.tp

    def negamax(self, state, color, depth=10):
        ''' implement negamax algorithm
        https://en.wikipedia.org/wiki/Negamax
        '''
        # negamax.counter += 1

        # CHECK LEAF NODE / DO NOT NEED TO CHECK DEPTH = 0 BECASE TicTacToe is too small
        # LEAF NODE is checked on play time

        # Transposition Table related work
        # ob = OptimalBoard(state)
        # _id = ob.board_id
        _id = OptimalBoard.board_to_id(state)

        cache = self.tp.get(_id)
        if cache is not None: # BUG FIX! cache can be 0, so should check None
            # case 1
            # return ob.convert_action_to_original(cache)
            return cache
            # case 2
            # return cache[0], random.choice(cache[1])

        # RECURSIVE
        actions = SP.available_actions(state)
        random.shuffle(actions) # move orders를 쓰면, alpha beta pruning시에 성능이 좋아짐
        best_score = -math.inf
        best_move = -1
        for action in actions:
            next_s = state[:]
            score, done = SP.play(next_s, action, color)
            if not done:
                score, _ = self.negamax(next_s, SP.next(color), depth-1)
                score = -score # negamax

            # pick from all best moves
            if score > best_score or (score == best_score and random.random() < 0.5):
                best_score = score
                best_move = action

        # case 1: choose random value 1 time
        # choosed_result = random.choice(best_scores)
        # tp.put(_id, choosed_result)
        # return choosed_result

        # case 2: choose random value every time
        self.tp.put(_id, (best_score, best_move))
        return (best_score, best_move)
