import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "lib"))

import tictactoe.gym as gym
import tictactoe.agent as agent
import tictactoe.utils as utils

import random

class QAgent(agent.AbstractAgent):
    def __init__(self, qlearning):
        super().__init__()
        self.q = qlearning
        self.mode = 0
        self.last_state = None
        self.last_action = None

    def set_mode(self, mode):
        '''mode=1: learning
        mode=0: real
        '''
        self.mode = mode

    def _reset(self):
        self.last_state = None
        self.last_action = None

    def _next_action(self, state, possible_actions):
        state = utils.compact_observation(state)
        if self.mode == 1:
            action = self.q.rargmax_with_exploit(state, possible_actions)
        else:
            action = self.q.rargmax(state, possible_actions)
        self.last_state = state
        self.last_action = action
        return action

    def feedback(self, next_state, reward):
        if self.last_state != None:
            next_state = utils.compact_observation(next_state)
            self.q.learn(self.last_state, self.last_action, reward, next_state)

