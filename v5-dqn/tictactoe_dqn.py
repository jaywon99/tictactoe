from collections import deque
import random
import numpy as np

from dqn import DQN

TRAINSET_SIZE = 100000
SAMPLE_SIZE = 64
DISCOUNT_RATE = 0.999
INPUT_SIZE = 27

FREQUENT_COPY=16
class TicTacToeDQN(DQN):
    def __init__(self, layers=None, lr=0.0001):
        super().__init__([27, 54, 54, 9], "main", lr)
        self.train_set = deque(maxlen=TRAINSET_SIZE)

    def state_converter(self, state):
        if INPUT_SIZE == 9:
            # MAKE 9
            return np.array(state[0])*state[1]

        elif INPUT_SIZE == 18:
            # (-1, 0, ...)을 각 셀당 두자리, 같으면 1, 다르면 0) - 총 18자리
            # return np.array(state)[:, 0]
            # state
            # print("STATE_CONVERTER", state)
            # MAKE 18 (-1, 1 only)
            board = [1 if x==state[1] else 0 for x in state[0]]
            board.extend([1 if x==-state[1] else 0 for x in state[0]])
            # print("board", board)
            return board

        elif INPUT_SIZE == 27:
            # MAKE 27 (0, 1, -1)
            board = [1 if x==0 else 0 for x in state[0]]
            board.extend([1 if x==state[1] else 0 for x in state[0]])
            board.extend([1 if x==-state[1] else 0 for x in state[0]])
            # print("board", board)
            return board

        else:
            raise Error

        ##############
        # logic for array
        a2 = np.array(state) # convert to nparray
        a3 = np.asarray([sublist for sublist in a2[:, 0]]) # fetch board tuple to list
        a4 = a2[:, 1:2] # fetch turn array
        board_my = (a3==a4).astype(int)
        board_other = (a3==-a4).astype(int)
        return np.concatenate((board_my, board_other),  axis=1)

    def predict_one(self, state):
        # print("PREDICT_ONE", state)
        return np.argmax(self.predict(self.state_converter(state)), axis=1)[0]
        # need reshape???

    def predict(self, state):
        return super().predict(state)

    def update(self, states, rewards):
        return super().update(states, rewards)

    def add_train_set(self, state, action, reward, next_state, done):
        # print("ADD_TRAIN_SET")
        self.train_set.append([self.state_converter(state), action, reward, self.state_converter(next_state), done])
        # print(self.train_set[-1])

    def study(self):
        if len(self.train_set) < SAMPLE_SIZE:
            return
        samples = random.sample(self.train_set, SAMPLE_SIZE)

        state_array = np.vstack([x[0] for x in samples])
        action_array = np.array([x[1] for x in samples])
        reward_array = np.array([x[2] for x in samples])
        next_state_array = np.vstack([x[3] for x in samples])
        done_array = np.array([x[4] for x in samples])

        X_batch = state_array
        y_batch = self.predict(state_array)
        Q_target = reward_array + DISCOUNT_RATE * np.max(self.predict(next_state_array), axis=1) * ~done_array
        y_batch[np.arange(len(X_batch)), action_array] = Q_target

        loss, _ = self.update(X_batch, y_batch)

        print("LOSS", loss)
        return loss

