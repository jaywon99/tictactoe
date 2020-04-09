import math

class OptimalBoard:
    # TOTAL 8 converion way
    CONVERSIONS = [
        [0, 1, 2, 3, 4, 5, 6, 7, 8], # -
        [0, 3, 6, 1, 4, 7, 2, 5, 8], # >y  (rotate 90 and flip y)
        [2, 1, 0, 5, 4, 3, 8, 7, 6], # y   (flip y)
        [2, 5, 8, 1, 4, 7, 0, 3, 6], # >>> (rotate 90 3 times)
        [6, 3, 0, 7, 4, 1, 8, 5, 2], # >   (rotate 90 1 time)
        [6, 7, 8, 3, 4, 5, 0, 1, 2], # >>y (rotate 90 2 times and flip y)
        [8, 5, 2, 7, 4, 1, 6, 3, 0], # >>>y (rotate 90 3 times and flip y)
        [8, 7, 6, 5, 4, 3, 2, 1, 0]  # >>  (rotate 90 2 times)
    ]

    @staticmethod
    def board_to_id(board):
        _id = 0
        for digit in (board):
            _id = (_id << 2) | (digit & 3)
        return _id

    def __init__(self, board):
        self.idx = -1
        self._board_id = math.inf
        for idx, C in enumerate(OptimalBoard.CONVERSIONS):
            y = [0] * 9
            for i, v in enumerate(board):
                y[C[i]] = v
            _id = OptimalBoard.board_to_id(y)
            if _id < self._board_id:
                self._board_id = _id
                self.idx = idx
                self._optimized_board = y

    @property
    def board_id(self):
        return self._board_id

    @property
    def optimal_board(self):
        return self._optimized_board

    def convert_action_to_optimal(self, actions):
        C = OptimalBoard.CONVERSIONS[self.idx]
        if isinstance(actions, list):
            return [C[action] for action in actions]
        else:
            return C[actions]

    def convert_action_to_original(self, action):
        C = OptimalBoard.CONVERSIONS[self.idx]
        for i, v in enumerate(C):
            if v == action:
                return i
        assert 0 == 1
