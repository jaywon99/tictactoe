from collections import deque
import random
import numpy as np

from dnn import DNN

TRAINSET_SIZE = 100000
SAMPLE_SIZE = 64
DISCOUNT_RATE = 0.999

FREQUENT_COPY=16
loss_counter = 0

class DQN(DNN):
    def __init__(self, layers=None, name="main", lr=0.0001):
        super().__init__(layers, name, lr)
        self.train_set = deque(maxlen=TRAINSET_SIZE)

    def predict_one(self, state):
        # print("PREDICT_ONE", state)
        return np.argmax(self.predict(state), axis=1)[0]

    def add_train_set(self, state, action, reward, next_state, done):
        # print("ADD_TRAIN_SET")
        self.train_set.append([state, action, reward, next_state, done])
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

        # Write using TeX (Basic Q-Learning)
        Q_target = reward_array + DISCOUNT_RATE * np.max(self.predict(next_state_array), axis=1) * ~done_array
        y_batch[np.arange(len(X_batch)), action_array] = Q_target

        loss, _ = self.update(X_batch, y_batch)

        global loss_counter
        loss_counter += 1
        if loss_counter > 100:
            loss_counter = 0
            print("LOSS", loss)
        return loss

