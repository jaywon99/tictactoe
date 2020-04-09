''' implement negamax algorithm '''

import random
import math
from collections import defaultdict

import pickle

from boardAI import AbstractPlayer, GameResult
from tictactoe import OptimalBoard, TicTacToeBoard
from tictactoe.selfplay import SelfPlayTicTacToeBoard as SP

class MCTSRandomPlayer(AbstractPlayer):
    ''' monte carlo tree search algoritm '''
    DEPTH = 10

    @staticmethod
    def _default_stats():
        return [0, 0]

    def __init__(self, simulations=10, C=1/math.sqrt(2), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.simulations = int(simulations)
        self.C = float(C)
        self._stats = defaultdict(MCTSRandomPlayer._default_stats)
        self.DEBUG = False

    def serialize(self):
        return pickle.dumps(self._stats)

    def deserialize(self, obj):
        if obj != None:
            self._stats = pickle.loads(obj)

    def _choose(self, state, available_actions):
        # return smart_turn(self.env)
        # state는 0,1,-1로 이루어진 9칸 array
        # 1. board를 만들 필요가 있을까? 아니면 그냥 state로 동작할까?
        # state, color, available_actions로 negamax를 동작
        # available_actions가 negamax에 필요한가? 그렇지 않음
        my_color = TicTacToeBoard.COLOR_TO_INTERNAL[self.color]
        stats, depth = self.search(state, my_color, self.simulations, self.C)
        return self.select_best_move(stats, depth, state, my_color)

    def search(self, board, start_color, simulations, C):
        ''' implement monte carlo tree search algorithm
        '''
        # CHECK LEAF NODE / DO NOT NEED TO CHECK DEPTH = 0 BECASE TicTacToe is too small
        # LEAF NODE is checked on play time

        stats = self._stats
        root = board
        max_depth = 0

        for _ in range(simulations):
            node = root[:]
            states = []

            # select leaf node
            depth = 0
            done = False
            color = start_color
            while not done:
                depth += 1
                action, select = self.select_next_move(stats, node, color, C)

                reward, done = SP.play(node, action, color)
                color = SP.next(color)

                states.append(OptimalBoard.board_to_id(node))

                if not select:
                    break

            max_depth = max(depth, max_depth)

            # run simulation if not at the end of the game tree
            if not done:
                result = self.simulate(node, start_color) # TODO: 여기를 어떻게 할지
            else:
                if reward == 0:
                    result = 0.5
                else:
                    result = 0

            # propagate results
            for state in reversed(states):
                result = 1 - result
                stats[state][0] += 1
                stats[state][1] += result

        return stats, max_depth

    def select_next_move(self, stats, board, color, C):
        """Select the next state and consider if it should be expanded"""

        bestscore = None
        bestmove = None

        children = []
        for action in SP.available_actions(board):
            next_board = board[:]
            SP.play(next_board, action, color)
            children.append((action, stats[OptimalBoard.board_to_id(next_board)]))

        total_n = sum(x[0] for (_, x) in children)

        for child_move, child_stat in children:
            n, w = child_stat
            if n == 0:
                return child_move, False
            else:
                score = (w / n) + C * math.sqrt(2 * math.log(total_n) / n)
                if bestscore is None or score > bestscore:
                    bestscore = score
                    bestmove = child_move

        assert bestmove is not None
        return bestmove, True        

    def select_best_move(self, stats, depth, board, color):
        """Select the best move at the end of the Monte Carlo tree search"""

        bestscore = 0
        bestmove = None
        total_n = 0

        for action in SP.available_actions(board):
            next_board = board[:]
            SP.play(next_board, action, color)
            n, w = stats[OptimalBoard.board_to_id(next_board)]
            total_n += n
            if self.DEBUG: print('Move %d score: %d/%d (%0.1f%%)' % (action, w, n, w/n*100))
            if n > bestscore or (n == bestscore and random.random() <= 0.5):
                bestmove = action
                bestscore = n
        assert bestmove is not None

        if self.DEBUG: print('Maximum depth: %d, Total simulations: %d' % (depth, total_n))

        return bestmove

    def simulate(self, board, start_color):
        # random simulator
        node = board[:]
        done = False
        color = start_color
        while not done:
            actions = SP.available_actions(node)

            reward, done = SP.play(node, random.choice(actions), color)
            color = SP.next(color)

        if reward == 0: # TIE
            return 0.5
        elif color == start_color:
            return 1
        else:
            return 0