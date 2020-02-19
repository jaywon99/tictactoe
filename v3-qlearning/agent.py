import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "lib"))

import tictactoe.gym as gym
import tictactoe.agent as agent
import tictactoe.utils as utils
from tictactoe.utils import OptimalBoard

import random

import qlearning

class MyAgent(agent.AbstractAgent):
    def __init__(self, learning_rate=0.1, discount_rate=0.9, exploit_rate=0.2):
        super().__init__()
        self.q = qlearning.qlearning(value=0, n_actions=9, learning_rate=learning_rate, discount=discount_rate, exploit_rate=exploit_rate)

    def save(self, filename):
        self.q.save(filename)

    def load(self, filename):
        self.q.load(filename)

    def _next_action(self, state, available_actions):
        ob = OptimalBoard(state)
        converted_actions = ob.convert_available_actions(available_actions)
        if self.train_mode:
            action = self.q.rargmax_with_exploit(ob.get_board_id(), converted_actions)
        else:
            action = self.q.rargmax(ob.get_board_id(), converted_actions)
        return ob.convert_to_original_action(action)

    def _feedback(self, state, action, next_state, reward, done):
        ob1 = OptimalBoard(state)
        compacted_state = ob1.get_board_id()
        ob2 = OptimalBoard(next_state)
        compacted_next_state = ob2.get_board_id()
        self.q.learn(ob1.get_board_id(), ob1.convert_available_action(action), reward, ob2.get_board_id())

