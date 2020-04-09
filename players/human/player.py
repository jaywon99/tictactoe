from boardAI import AbstractPlayer
from tictactoe import OptimalBoard

class WrongMoveError(Exception):
    pass

class HumanPlayer(AbstractPlayer):
    ''' random player '''
    def _choose(self, state, available_actions):

        self.render_for_human(state)
        while True:
            try:
                move = int(input('Your move: ')) - 1
                if move not in available_actions:
                    raise WrongMoveError('available only '+','.join([str(i) for i in available_actions]))
            except ValueError:
                print('Wrong move! Must be an integer between 1-9.')
            except WrongMoveError as e:
                print(e)
            else:
                break

        return move

    def render_for_human(self, state):
        ALL_COLORS = [' ', 'O', 'X']    # 0, 1, 2

        print('---+---+----')
        for row in range(0, 9, 3):
            for idx in range(row, row+3):
                print('', ALL_COLORS[state[idx]], '|', end='')
            print('')
            print('---+---+----')
        print('-------------------------------')        