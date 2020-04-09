from .board import TicTacToeBoard

# TODO: SelfPlay와 TicTacToeBoard를 합칠 방법을 찾아보자!
class SelfPlayTicTacToeBoard:
    @staticmethod
    def play(board, pos, color):
        board[pos] = color
        space_left = board.count(0)
        reward = SelfPlayTicTacToeBoard.check_win(board, pos, color)
        if reward:
            return space_left+1, True
        elif space_left == 0:
            return space_left, True
        else:
            return 0, False 

    @staticmethod
    def winning_move(board, pos, color):
        assert(board[pos]==0)
        board[pos] = color
        result = SelfPlayTicTacToeBoard.check_win(board, pos, color)
        board[pos] = 0
        return result

    @staticmethod
    def available_actions(board):
        return [i for i, v in enumerate(board) if v == 0]

    @staticmethod
    def next(color):
        return -color

    @staticmethod
    def check_win(board, pos, color):
        # return reward, done
        for line in TicTacToeBoard.WINNING_LINES:
            # if this action is not in checking line, pass!
            if pos not in line:
                continue

            if board[line[0]] == color and \
               board[line[1]] == color and \
               board[line[2]] == color:
                # game finished and winner is color
                return True

        return False

            
