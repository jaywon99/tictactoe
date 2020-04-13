''' DNN module '''

import numpy as np

import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Flatten, BatchNormalization
from keras.optimizers import Adam
from keras.utils import multi_gpu_model

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
        self.model = Sequential()
        self.model.add(Dense(self.layers[1], input_shape=(self.layers[0],)))
        self.model.add(Activation('relu'))
        for i in range(2, len(self.layers)-1):
            self.model.add(Dense(self.layers[i]))
            self.model.add(Activation('relu'))
        self.model.add(Dense(self.layers[-1]))

        # # self.loss = tf.losses.softmax_cross_entropy(self.Y, self.Q)
        # # self.train = tf.train.RMSPropOptimizer(self.learning_rate, decay=0.75).minimize(self.loss)
        self.model.compile(loss=keras.losses.mean_squared_error, optimizer=Adam(
            learning_rate=self.learning_rate), metrics=['accuracy'])

    def predict(self, state):
        ''' get prediction result of state '''
        input_values = np.reshape(state, [-1, self.layers[0]])
        result = self.model.predict(input_values)
        return result

    def update(self, x_stack, y_stack):
        ''' dnn update '''
        hist = self.model.fit(x=x_stack, y=y_stack, epochs=10, verbose=0)
        return hist.history['loss'], None

    def save(self, filename):
        ''' save network variables '''
        self.model.save(filename+".h5")

    def load(self, filename):
        ''' load network variables '''
        try:
            self.model = load_model(filename+".h5")
        except Exception:
            print("Loading Exception")  # loading error - ignore

