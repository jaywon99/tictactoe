import pickle

# 문제1: 총 623529개의 case가 존재
# 이것을 O/X/ 의 9개짜리로 변경

class PredictionTable:
    def __init__(self, learning_rate = 0.1):
        self.pred_table = {}
        self.learning_rate = learning_rate
        self.step = 0

    def zip_board(self, board):
        filled = [int(s) for s in board]
        maps = [' '] * 9
        for i in range(len(filled)):
            if i % 2 == 1:
                maps[filled[i]] = 'O'
            else:
                maps[filled[i]] = 'X'
        return ''.join(maps)

    def lookup(self, board):
        maps = self.zip_board(board)
        # print(maps)
        if maps not in self.pred_table:
            self.pred_table[maps] = 0.5
        return self.pred_table[maps]

    def learn(self, board, next_predict):
        maps = self.zip_board(board)
        p = 0.5 if maps not in self.pred_table else self.pred_table[maps]
        next_predict = p + self.learning_rate * (next_predict - p)
        self.pred_table[maps] = next_predict
        return next_predict

    def set(self, board, value):
        maps = self.zip_board(board)
        self.pred_table[maps] = value
        return value

    def next_step(self):
        self.step += 1

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump((self.step, self.learning_rate, self.pred_table), f)

    def load(self, filename):
        try:
            with open(filename, 'rb') as f:
                self.step, self.learing_rate, self.pred_table = pickle.load(f)
        except:
            self.pred_table = {}
            # self.learning_rate = learning_rate
            self.step = 0

def learning(p_table, board, winner):
    # for winner
    board_winner = board[:]
    np = 1.0 if winner != '=' else 0.5
    p_table.set(board_winner, np)
    # print("RESULT", board_winner, np)
    while len(board_winner) > 1:
        board_winner = board_winner[:-2]
        np = p_table.learn(board_winner, np)
        # print("LEARN", board_winner, np)
        # print(board_winner, p, np)

    board_looser = board[:-1]
    np = 0.0 if winner != '=' else 0.5
    p_table.set(board_looser, np)
    # print("RESULT", board_looser, np)
    while len(board_looser) > 1:
        board_looser = board_looser[:-2]
        np = p_table.learn(board_looser, np)
        # print("LEARN", board_looser, np)
        # print(board_looser, p, np)

    # print("LEARNED------------")
