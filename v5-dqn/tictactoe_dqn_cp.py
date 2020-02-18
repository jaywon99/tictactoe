from collections import deque
import random
import numpy as np
import tensorflow as tf

from dqn import DQN

TARGET_UPDATE_FREQUENCY = 5
TRAINSET_SIZE = 100000
SAMPLE_SIZE = 64
DISCOUNT_RATE = 0.999
INPUT_SIZE = 27

FREQUENT_COPY=16
class TicTacToeDQN():
    def __init__(self, layers=None, lr=0.0001):
        # super().__init__([18, 54, 54, 9], name, lr)
        self.main = DQN([INPUT_SIZE, 27, 18, 9], "main", lr)
        self.target = DQN([INPUT_SIZE, 27, 18, 9], "target", lr)
        self.train_set = deque(maxlen=TRAINSET_SIZE)
        self.copy_dqn = self.get_copy_var_ops(self.target.net_name, self.main.net_name)
        self.study_counter = 0

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
        return self.main.predict(state)

    def update(self, states, rewards):
        return self.main.update(states, rewards)

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
        # You can use tf.where
        Q_target = reward_array + DISCOUNT_RATE * np.max(self.target.predict(next_state_array), axis=1) * ~done_array
        y_batch[np.arange(len(X_batch)), action_array] = Q_target

        loss, _ = self.update(X_batch, y_batch)

        self.study_counter += 1
        if self.study_counter % TARGET_UPDATE_FREQUENCY == 0:
            self.copy()

        print("LOSS", loss)
        return loss

    def copy(self):
        return self.session.run(self.copy_dqn)

    # main to target
    def get_copy_var_ops(self, dest_scope_name, src_scope_name):
        # Copy variables src_scope to dest_scope
        op_holder = []

        src_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope=src_scope_name)
        dest_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope=dest_scope_name)

        for src_var, dest_var in zip(src_vars, dest_vars):
            op_holder.append(dest_var.assign(src_var.value()))

        return op_holder

    def save(self, filename):
        self.main.save(filename)

    def load(self, filename):
        self.main.load(filename)

    def set_session(self, session):
        self.session = session
        self.main.set_session(session)
        self.target.set_session(session)

