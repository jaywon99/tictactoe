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
    COLORS = ['X', 'O']

    def __init__(self):
        self.init_board()

    def init_board(self):
        self.seq = ""

    def seq_to_board(self, seq=None, symbol=[' ', 'X', 'O']):
        seq = self.seq if seq == None else seq
        filled = [int(s) for s in seq]
        board = [symbol[0]] * 9
        for i in range(len(filled)):
            if i % 2 == 0:
                board[filled[i]] = symbol[1]
            else:
                board[filled[i]] = symbol[2]
        return board

    # NEED TO OPTIMIZE
    # convert to number and -1
    def _alternate_ids(self, board):
        # 123 369 987 741 321 963 789 147
        # 456 258 654 852 654 852 456 258
        # 789 147 321 963 987 741 123 369
        convs = ['123456789',
                 '369258147',
                 '987654321',
                 '741852963',
                 '321654987',
                 '963852741',
                 '789456123',
                 '147258369']

        return [ [board[int(i)-1] for i in c] for c in convs ]

    def get_board_id(self, seq=None):
        board = self.seq_to_board(seq, symbol=[0,1,2])

        alternates = self._alternate_ids(board)

        _id = 4 << 18
        for alternate in alternates:
            _alt_id = 0
            for digit in alternate:
                _alt_id = (_alt_id << 2) | digit
            if _alt_id < _id:
                _id = _alt_id
        return _id

    def get_raw_board_id(self, seq=None):
        board = self.seq_to_board(seq, symbol=[0,1,2])

        alternates = self._alternate_ids(board)
        _id = 0
        for digit in alternates[0]:
            _id = (_id << 2) | digit
        return _id

    def get_next_board_id(self, pos):
        return self.get_board_id(self.seq+str(pos))

    def get_candidates(self):
        full = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        filled = [int(s) for s in self.seq]
        return list(set(full) - set(filled))

    def print_board(self):
        board = self.seq_to_board()
        print('---+---+----')
        for y in range(0, 9, 3):
            for x in range(y, y+3):
                print('', board[x], '|', end='')
            print('')
            print('---+---+----')
        print('-------------------------------')

    def is_win_board(self, board):
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
            if board[line[0]] == board[line[1]] and board[line[1]] == board[line[2]] and board[line[0]] != ' ':
                return board[line[0]]

        if ' ' in board:
            return ' '
        else:
            return '='

    def is_win(self):
        board = self.seq_to_board()
        return self.is_win_board(board)

    def play(self, pos):
        self.seq += str(pos)
        return self.is_win()

    def unplay(self):
        self.seq = self.seq[:-1]

    def turn_color(self):
        return TicTacToeBoard.COLORS[len(self.seq) % 2]

    def other_color(self):
        return TicTacToeBoard.COLORS[(len(self.seq)+1) % 2]

    def can_win(self, pos, color): # color is O or X
        board = self.seq_to_board()
        board[pos] = color
        rt = self.is_win_board(board)
        board[pos] = ' '    # restore to empty cell
        return rt == color


