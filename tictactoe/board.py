import math
from enum import Enum

from boardAI import AbstractBoard, GameResult

# TODO: 자료 형식을 정의하자 (1. internal, 2. between player, 3. rendering)
class TicTacToeBoard(AbstractBoard):
    WINNING_LINES = [[0, 1, 2],
                     [3, 4, 5],
                     [6, 7, 8],
                     [0, 3, 6],
                     [1, 4, 7],
                     [2, 5, 8],
                     [0, 4, 8],
                     [2, 4, 6]]

    ALL_COLORS = {0:' ', 1:'O', -1:'X'}    # 0, 1, -1
    COLORS = ['O', 'X']
    COLOR_TO_INTERNAL = {'O': 1, 'X': -1}

    class Reward:
        WIN = 1
        TIE = 0
        PLAYING = 0
        ERROR = -math.inf

    def __init__(self):
        super().__init__()

    def reset(self):
        super().reset()
        # 0 means EMPTY, 1 means Player #1, -1 means Player #2
        self._board = [0]*9
        self._n_turn = 0

    def get_colors(self):
        return TicTacToeBoard.COLORS

    @property
    def n_turn(self):
        return self._n_turn

    def _render_for_human(self):
        print('---+---+----')
        for row in range(0, 9, 3):
            for idx in range(row, row+3):
                print('', TicTacToeBoard.ALL_COLORS[self._board[idx]], '|', end='')
            print('')
            print('---+---+----')
        print('-------------------------------')

    def render(self, mode='human'):
        ''' display board '''
        if mode == 'human':
            return self._render_for_human()
        # if mode == 'network' or etc
        return

    def get_status(self, color):
        ''' return status by color and available actions. if cell in board == 0, it's available. '''
        return self.to_status(color), [i for i, v in enumerate(self._board) if v == 0]

    def play(self, action, color):
        ''' play action and get result.
        reward (if possible, depend on game), result (GameResult - PLAY_NEXT, PLAY_AGAIN, END_WIN, END_TIE, END_ERROR)
        '''
        if self.end:
            # SHOULD NOT BE HERE! EXCEPTION vs ERROR MESSAGE vs LOOSE MESSAGE
            return TicTacToeBoard.Reward.ERROR, GameResult.END, None

        if not self.is_possible_action(action):
            # SHOULD NOT BE HERE! EXCEPTION vs ERROR MESSAGE vs LOOSE MESSAGE
            # status, reward (-math.inf), END
            return TicTacToeBoard.Reward.ERROR, GameResult.END, None

        self._n_turn += 1
        self._board[action] = TicTacToeBoard.COLOR_TO_INTERNAL[color]
        won = self.win(action, color)

        if won:
            self.end = True
            return TicTacToeBoard.Reward.WIN, GameResult.END, None

        if 0 in self._board:  # still place to put stone
            # We don't know score yet.
            return TicTacToeBoard.Reward.PLAYING, GameResult.PLAY_NEXT, None

        self.end = True
        return TicTacToeBoard.Reward.TIE, GameResult.END, None

    def is_possible_action(self, action):
        return self._board[action] == 0

    def is_winning_move(self, action, color):
        if self.end:
            return False

        if not self.is_possible_action(action):
            return False

        self._board[action] = TicTacToeBoard.COLOR_TO_INTERNAL[color]
        result = self.win(action, color)
        self._board[action] = 0
        return result

    def win(self, action, color):
        ''' evaluate board and return result
        '''
        internal_color = TicTacToeBoard.COLOR_TO_INTERNAL[color]
        for line in TicTacToeBoard.WINNING_LINES:
            # if this action is not in checking line, pass!
            if action not in line:
                continue

            if self._board[line[0]] == internal_color and \
               self._board[line[1]] == internal_color and \
               self._board[line[2]] == internal_color:
                # game finished and winner is color
                return True

        return False

    def to_status(self, color):
        # TODO: color 입장에서 board를 다시 보여주기 (어떻게???)
        return self._board[:]
