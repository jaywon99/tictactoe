import random
import pickle

MARKER = {-1: 'O', 0: ' ', 1: 'X'}
class TicTacToeBoard:
    def __init__(self):
        self.reset()

    def reset(self):
        self.board = [0]*9  # 0 means EMPTY, 1 means Player #1, -1 means Player #2
        self.done = False   # Game Done

    def is_game_finished(self):
        return self.done

    def available_actions(self):
        return [i for i, v in enumerate(self.board) if v == 0]

    def is_possible_action(self, action):
        return self.board[action] == 0

    def step(self, action, turn):
        assert self.is_possible_action(action)

        self.board[action] = turn
        return self.check_game_status()

    def check_game_status(self):
        ''' Return (Finished, Winner)
        '''
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
            win = self.board[line[0]] + self.board[line[1]] + self.board[line[2]]
            if win !=0 and win % 3 == 0:
                # game finished and winner is win//3
                self.done = True
                return True, win//3     

        if 0 in self.board:
            # in game
            return False, 0
        # game finished and tie
        self.done = True
        return True, 0

    def render(self, mode='human'):
        print('---+---+----')
        for row in range(0, 9, 3):
            for idx in range(row, row+3):
                print('', MARKER[self.board[idx]], '|', end='')
            print('')
            print('---+---+----')
        print('-------------------------------')

    def create_memento(self):
        return pickle.dumps(self.board)

    def set_memento(self, memento):
        self.board = pickle.loads(memento)
        self.done, _ = self.check_game_status()


class TicTacToeAction:
    def __init__(self, board):
        self.n = 9
        self.board = board

    def sample(self):
        return random.choice(self.board.available_actions())

    def contains(self, action):
        return action in self.board.available_actions()

class TicTacToeObservation:
    def __init__(self, board):
        self.shape = (1,)
        self.board = board

    def _get_obs(self, turn):
        return [cell*turn for cell in self.board.board] # or mulpiply turn to each board cells

# BOARD: 0 = EMPTY, 1 = X, 2 = O
class TicTacToeEnv:
    def __init__(self):
        self.board = TicTacToeBoard()
        self.action_space = TicTacToeAction(self.board)
        self.observation_space = TicTacToeObservation(self.board)
        self.reward_range = (-1, 1)
        self.metadata = {'render.modes': []}
        self.turn = 1

    @classmethod
    def class_name(cls):
        return cls.__name__

    def reset(self):
        '''
            Reset TicTacToe Environment. Restart all game, player, turn, etc
        '''
        self.board.reset()
        return self.observation_space._get_obs(self.turn)

    def close(self):
        pass # or destroy classes

    def seed(self, seed):
        pass

    def render(self, mode='human'):
        self.board.render()

    def available_actions(self):
        return self.board.available_actions()

    def step(self, action):
        if self.board.is_game_finished():
            return self.observation_space._get_obs(self.board, self.turn), 0, True, None

        done, reward = self.board.step(action, self.turn)

        self.turn = -self.turn # TURN: -1 <-> 1
        return self.observation_space._get_obs(self.turn), reward, done, None
        # return (observation, reward, done, info)

    def create_memento(self):
        return pickle.dumps((self.turn, self.board.create_memento()))

    def set_memento(self, memento):
        self.turn, board_memento = pickle.loads(memento)
        self.board.set_memento(board_memento)


