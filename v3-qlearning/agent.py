import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "lib"))

import tictactoe.gym as gym
import tictactoe.agent as agent
import tictactoe.utils as utils

import random

import qlearning

class MyAgent(agent.AbstractAgent):
    def __init__(self, learning_rate=0.1, discount_rate=0.9, exploit_rate=0.2):
        super().__init__()
        self.q = qlearning.qlearning(value=0, n_actions=9, learning_rate=learning_rate, discount=discount_rate, exploit_rate=exploit_rate)
        self.mode = 0
        self.last_state = None
        self.last_action = None

    def save(self, filename):
        self.q.save(filename)

    def load(self, filename):
        self.q.load(filename)

    def _reset(self, feedback, episode=-1):
        self.last_state = None
        self.last_action = None

    def _next_action(self, state, available_actions):
        state = utils.compact_observation(state)
        if self.train_mode:
            action = self.q.rargmax_with_exploit(state, available_actions)
        else:
            action = self.q.rargmax(state, available_actions)
        self.last_state = state
        self.last_action = action
        return action

    def feedback(self, next_state, reward, done):
        if self.last_state != None:
            next_state = utils.compact_observation(next_state)
            self.q.learn(self.last_state, self.last_action, reward, next_state)

