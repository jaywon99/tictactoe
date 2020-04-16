''' qlearning agent '''
import random
import pickle

from boardAI import AbstractPlayer
from tictactoe import TicTacToeBoard
from tictactoe.selfplay import SelfPlayTicTacToeBoard as SP

class RandomPlusPlayer(AbstractPlayer):
    ''' random player, but in winning move, play first. '''
    def _choose(self, state, available_actions):
        my_color = TicTacToeBoard.COLOR_TO_INTERNAL[self.color]
        opposite_color = SP.next(my_color)
        for action in available_actions:
            if SP.winning_move(state, action, my_color):
                # is this my winning move?
                return action
            if SP.winning_move(state, action, opposite_color):
                # is this opposite winning move?
                return action
        return random.choice(available_actions)
