import random

# TIC TAC TOE BOARD
# 1 | 2 | 3
# ---------
# 4 | 5 | 6
# ---------
# 7 | 8 | 9
#
# 여기서 여짓것 play한 위치 (1~9)를 순서로 넣는다.
# X PLAY FIRST

class TicTacToeBoard:
    def __init__(self):
        self.board = self.init_board()

    def init_board(self):
        return ""

    def get_candidate(self, board):
        full = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        filled = [int(s) for s in board]
        return list(set(full) - set(filled))

    def board_to_map(self, board):
        filled = [int(s) for s in board]
        maps = [' '] * 9
        for i in range(len(filled)):
            if i % 2 == 1:
                maps[filled[i]] = 'O'
            else:
                maps[filled[i]] = 'X'
        return maps

    def print_board(self, board):
        maps = self.board_to_map(board)
        print('---+---+----')
        for y in range(0, 9, 3):
            for x in range(y, y+3):
                print('', maps[x], '|', end='')
            print('')
            print('---+---+----')
        print('-------------------------------')

    def is_win(self, board):
        if len(board) == 9:
            return '='

        maps = self.board_to_map(board)
        checking = [
                [0, 1, 2],
                [3, 4, 5],
                [6, 7, 8],
                [0, 3, 6],
                [1, 4, 7],
                [2, 5, 8],
                [0, 4, 8],
                [2, 4, 6]]
        for line in checking:
            if maps[line[0]] == maps[line[1]] and maps[line[1]] == maps[line[2]] and maps[line[0]] != ' ':
                return maps[line[0]]

        return ' '
