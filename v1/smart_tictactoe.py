import random

from tictactoe import TicTacToeBoard
from ptable import PredictionTable 

class AutoTicTacToe(TicTacToeBoard):
    def __init__(self, **kwargs):
        super().__init__()
        self.debug = kwargs['debug']

    def play_random(self, board):
        candidate = self.get_candidate(board)
        random.shuffle(candidate)
        return board + str(candidate[0])

    def play_o(self, board):
        return self.play_random(board)

    def play_x(self, board):
        return self.play_random(board)

    def play(self, board):
        if len(board) % 2 == 0:
            return self.play_x(board)
        else:
            return self.play_o(board)

    def play_game(self):
        board = self.init_board()
        # print_board(board[-1])
        winner = self.is_win(board)
        while winner == ' ':
            board = self.play(board)
            winner = self.is_win(board)
            # print_board(board)
        return (board, winner)

class SmartTicTacToe(AutoTicTacToe):
    def __init__(self, p_table, **kwargs):
        super().__init__(**kwargs)
        self.p_table = p_table

    def play_smart(self, board):
        candidates = self.get_candidate(board)
        found_p = -1.0
        found_c = -1
        if self.debug: print("FROM", board)
        for candidate in candidates:
            next_board = board + str(candidate)
            p = self.p_table.lookup(next_board)
            if self.debug: print("CANDIDATE", candidate, p)
            if p > found_p:
                found_p = p
                found_c = candidate
        if self.debug: print("SELECT", found_c, found_p)
        return board + str(found_c)

    def play(self, board):
        if len(board) % 2 == 0:
            next_board = self.play_x(board)
        else:
            next_board = self.play_o(board)
        if self.debug: self.print_board(next_board)
        return next_board


class SmartO(SmartTicTacToe):
    def play_o(self, board):
        return self.play_smart(board)

class SmartX(SmartTicTacToe):
    def play_x(self, board):
        return self.play_smart(board)

class SmartOX(SmartTicTacToe):
    def play_o(self, board):
        return self.play_smart(board)

    def play_x(self, board):
        return self.play_smart(board)

