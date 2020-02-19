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
        self.dqn = dqn
        self.egreedy = egreedy

    def _next_action(self, state, available_actions):
        ###
        if self.train_mode:
            if random.random() < self.egreedy:
                action = random.choice(available_actions)
            else:
                action = self.dqn.predict_one(state)
        else:
            action = self.dqn.predict_one(state)

        if action not in available_actions:
            self.dqn.add_train_set(state, action, -1, ((0,0,0,0,0,0,0,0,0),self.turn), 1)
            action = random.choice(available_actions)

        return action

    def _feedback(self, state, action, next_state, reward, done):
        self.dqn.add_train_set(state, action, reward, next_state, done)
        self.dqn.study()

