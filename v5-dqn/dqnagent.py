import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "lib"))

import tictactoe.gym as gym
import tictactoe.agent as agent
import tictactoe.utils as utils

import random

class DQNAgent(agent.AbstractAgent):
    def __init__(self, turn, dqn, egreedy=0.2):
        super().__init__()
        self.mode = 0
        self.turn = turn
        self.last_state = None
        self.last_action = None
        self.dqn = dqn
        self.egreedy = egreedy

    def set_mode(self, mode):
        self.mode = mode

    def _reset(self):
        self.last_state = None
        self.last_action = None

    def _next_action(self, state, available_actions):
        ###
        if self.mode == 1:
            if random.random() < self.egreedy:
                action = random.choice(available_actions)
            else:
                action = self.dqn.predict_one(state)
        else:
            action = self.dqn.predict_one(state)
        if action not in available_actions:
            self.dqn.add_train_set(state, action, -1, ((0,0,0,0,0,0,0,0,0),self.turn), 1)
            action = random.choice(available_actions)
        self.last_state = state
        self.last_action = action
        return action

    def feedback(self, next_state, reward, done):
        if self.last_action != None:
            self.dqn.add_train_set(self.last_state, self.last_action, reward, next_state, done)
            self.dqn.study()

