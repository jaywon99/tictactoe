''' DNN module '''

import numpy as np
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

class DNN:
    ''' DNN class '''
    def __init__(self, layers, name="main", learning_rate=0.001):
        self.layers = layers
        self.net_name = name
        self.learning_rate = learning_rate
        self.session = None

        self._build_network()

    def _build_network(self):
        ''' build DNN network '''
        self.X = tf.placeholder(tf.float32, [None, self.layers[0]], name="input")
        self.Y = tf.placeholder(tf.float32, [None, self.layers[-1]], name="output")
        net = self.X
        for i in range(1, len(self.layers)-1):
            net = tf.layers.dense(net, self.layers[i], activation=tf.nn.relu)
        self.Q = tf.layers.dense(net, self.layers[-1])

        # self.loss = tf.losses.softmax_cross_entropy(self.Y, self.Q)
        self.loss = tf.losses.mean_squared_error(self.Y, self.Q)
        self.train = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        # self.train = tf.train.RMSPropOptimizer(self.learning_rate, decay=0.75).minimize(self.loss)

    def set_session(self, session):
        ''' set tensorflow session '''
        self.session = session

    def predict(self, state):
        ''' get prediction result of state '''
        input_values = np.reshape(state, [-1, self.layers[0]])
        return self.session.run(self.Q, feed_dict={self.X: input_values})

    def update(self, x_stack, y_stack):
        ''' dnn update '''
        self.session.run([self.loss, self.train], {self.X: x_stack, self.Y: y_stack})
        return self.session.run([self.loss, self.train], {self.X: x_stack, self.Y: y_stack})

    def save(self, filename):
        ''' save network variables '''
        saver = tf.train.Saver()
        saver.save(self.session, filename)

    def load(self, filename):
        ''' load network variables '''
        try:
            saver = tf.train.Saver()
            saver.restore(self.session, filename)
        except Exception:
            print("Loading Exception") # loading error - ignore
