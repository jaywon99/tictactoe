from tictactoe import OptimalBoard as OB
from tictactoe.selfplay import SelfPlayTicTacToeBoard as SP

MARKER={-1:'O', 0:'=', 1:'X'}
def find_next(board, color, seq):
    actions = SP.available_actions(board)

    for action in actions:
        new_board = board[:]
        reward, done = SP.play(new_board, action, color)
        if done == True:
            # print it?
            if reward == 0:
                print(seq+str(action), '=', OB.board_to_id(new_board))
            else:
                print(seq+str(action), MARKER[color], OB.board_to_id(new_board))
        else:
            find_next(new_board, SP.next(color), seq+str(action))

board = [0]*9
color = 1
find_next(board, color, '')

