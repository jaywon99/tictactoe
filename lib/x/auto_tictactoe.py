from tictactoe import TicTacToeBoard

class AutoTicTacToe(TicTacToeBoard):
    def __init__(self, player1, player2, **kwargs):
        super().__init__()
        self.player1 = player1
        self.player2 = player2
        self.debug = kwargs['debug'] if 'debug' in kwargs else False

    def turn(self):
        if len(self.seq) % 2 == 0:
            return self.player1.play(self)
        else:
            return self.player2.play(self)

    def auto_game(self):
        # print_board(board[-1])
        winner = self.is_win()
        while winner == ' ':
            if self.debug: self.print_board()
            winner = self.turn()
        return winner

