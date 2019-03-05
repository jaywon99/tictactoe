import random
import pickle

ALL_ACTIONS = [0,1,2,3,4,5,6,7,8]
class TicTacToeAction:
    def __init__(self):
        self.n = 9
        self.space = ()
        self.board = None

    def set_board(self, board):
        self.board = board

    def sample(self):
        if self.board:
            return random.choice(self.board.available_actions())
        return random.randint(0, self.n-1)

    def contains(self, action):
        if self.board:
            return action in self.board.available_actions()
        return action in ALL_ACTIONS

class TicTacToeObservation:
    def __init__(self):
        self.shape = (1,)
        self.board = None

    def _get_obs(self, board, turn):
        return tuple(board), turn

def check_game_status(board):
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
        win = board[line[0]] + board[line[1]] + board[line[2]]
        # if board[line[0]] == board[line[1]] and board[line[1]] == board[line[2]] and board[line[0]] != 0:
        if win == 3 or win == -3: # X win
            return True, win//3

    if 0 in board:
        return False, 0
    return True, 0

# BOARD: 0 = NOTHING, 1 = X, 2 = O
MARKER = {-1: 'O', 0: ' ', 1: 'X'}
class TicTacToeEnv:
    def __init__(self):
        self.action_space = TicTacToeAction()
        self.observation_space = TicTacToeObservation()
        self.reward_range = (-1, 1)
        self.metadata = {'render.modes': []}
        pass

    @classmethod
    def class_name(cls):
        return cls.__name__

    def reset(self):
        self.board = [0]*9
        self.turn = 1
        self.done = False
        self.action_space.set_board(self)
        return self.observation_space._get_obs(self.board, self.turn)

    def close(self):
        self.action_space.set_board(None)

    def seed(self, seed):
        pass

    def render(self, mode='human'):
        print('---+---+----')
        for y in range(0, 9, 3):
            for x in range(y, y+3):
                print('', MARKER[self.board[x]], '|', end='')
            print('')
            print('---+---+----')
        print('-------------------------------')

    def available_actions(self):
        return [i for i, v in enumerate(self.board) if v == 0]

    def step(self, action):
        assert self.action_space.contains(action)

        if self.done:
            return self.observation_space._get_obs(self.board, self.turn), 0, True, None

        self.board[action] = self.turn
        done, reward = check_game_status(self.board)

        self.turn = -self.turn
        return self.observation_space._get_obs(self.board, self.turn), reward, done, None
        # return (observation, reward, done, info)

    # eclose, compute_reward, metadata, reward_range, seed, spec
    def create_memento(self):
        return pickle.dumps((self.turn, self.done, self.board))

    def set_memento(self, memento):
        self.turn, self.done, self.board = pickle.loads(memento)


