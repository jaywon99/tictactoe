import numpy as np
import tensorflow as tf

class DQN:
    def __init__(self, layers, name="main", lr=0.001):
        self.layers = layers
        self.net_name = name
        self.lr = lr

        self._build_network()

    def _build_network(self):
        self._X = tf.placeholder(tf.float32, [None, self.layers[0]], name="input")
        self._Y = tf.placeholder(tf.float32, [None, self.layers[-1]], name="output")
        net = self._X
        for i in range(1, len(self.layers)-1):
            net = tf.layers.dense(net, self.layers[i], activation=tf.nn.tanh)
        self._Q = tf.layers.dense(net, self.layers[-1])

        # self._loss = tf.losses.softmax_cross_entropy(self._Y, self._Q)
        self._loss = tf.losses.mean_squared_error(self._Y, self._Q)
        self._train = tf.train.AdamOptimizer(self.lr).minimize(self._loss)
        # self._train = tf.train.RMSPropOptimizer(self.lr, decay=0.75).minimize(self._loss)

    def set_session(self, session):
        self.session = session

    def predict(self, state):
        x = np.reshape(state, [-1, self.layers[0]])
        return self.session.run(self._Q, feed_dict={self._X: x})

    def update(self, x_stack, y_stack):
        self.session.run([self._loss, self._train], {self._X: x_stack, self._Y: y_stack})
        self.session.run([self._loss, self._train], {self._X: x_stack, self._Y: y_stack})
        self.session.run([self._loss, self._train], {self._X: x_stack, self._Y: y_stack})
        self.session.run([self._loss, self._train], {self._X: x_stack, self._Y: y_stack})
        return self.session.run([self._loss, self._train], {self._X: x_stack, self._Y: y_stack})

    def save(self, filename):
        saver = tf.train.Saver()
        saver.save(self.session, filename)

    def load(self, filename):
        saver = tf.train.Saver()
        saver.restore(self.session, filename)

