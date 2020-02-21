from collections import deque
import random
import numpy as np
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from dqn import DQN

TARGET_UPDATE_FREQUENCY = 16
TRAINSET_SIZE = 100000
SAMPLE_SIZE = 64
DISCOUNT_RATE = 0.95

FREQUENT_COPY=16
class DDQN():
    def __init__(self, layers=None, lr=0.001):
        # super().__init__([18, 54, 54, 9], name, lr)
        self.main = DQN(layers, "main", lr)
        self.target = DQN(layers, "target", lr)
        self.train_set = deque(maxlen=TRAINSET_SIZE)
        self.copy_dqn = self.get_copy_var_ops(self.target.net_name, self.main.net_name)
        self.study_counter = 0

    def predict_one(self, state):
        # print("PREDICT_ONE", state)
        return np.argmax(self.predict(state), axis=1)[0]
        # need reshape???

    def predict(self, state):
        return self.main.predict(state)

    def predict_by_target(self, state, actions):
        predict = self.target.predict(state)
        return predict[np.arange(len(predict)), actions]

    def update(self, states, rewards):
        return self.main.update(states, rewards)

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

        # print(action_array)
        X_batch = state_array
        # print(X_batch)
        y_batch = self.predict(state_array)
        # print(y_batch)
        # Write using TeX
        # s_t = state_array
        # s_t1 = next_state_array
        # Q(s_t, *) = self.predict(state_array)
        # Q(s_t1, *) = self.predict(next_state_array)
        # selecting the best action a with maximum Q-value of next state.
        # argmax(Q(s_t1, *)) = np.argmax(self.predict(next_state_array))
        next_predict = self.predict(next_state_array)
        # print(next_predict)
        next_best_action = np.argmax(next_predict, axis=1)  # 이걸 row/col을 바꿔야 할지..
        # print(next_best_action)
        # calculating expected Q-value by using the action a selected above.
        q_estimated = self.predict_by_target(next_state_array, next_best_action)
        # print(q_estimated)
        Q_target = reward_array + DISCOUNT_RATE * q_estimated * ~done_array
        # print(Q_target)
        y_batch[np.arange(len(X_batch)), action_array] = Q_target
        # print(y_batch)

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
        self.copy()

