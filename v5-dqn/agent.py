import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "lib"))

import tictactoe.gym as gym
import tictactoe.agent as agent
import tictactoe.utils as utils
from tictactoe.utils import OptimalBoard

import random
import numpy as np

from dqn import DQN

INPUT_SIZE = 18
class MyAgent(agent.AbstractAgent):
    def __init__(self, egreedy=0.2):
        super().__init__()
        self.mode = 0
        self.dqn = DQN(layers=[INPUT_SIZE, 54, 54, 9], lr=0.001)
        self.egreedy = egreedy

    def save(self, filename):
        self.dqn.save(filename)

    def load(self, filename):
        self.dqn.load(filename)

    def set_session(self, session):
        self.dqn.set_session(session)

    def state_converter(self, state):
        if INPUT_SIZE == 9:
            # MAKE 9 / (-1, 0, 1)로 되어있는 board state
            return state

        elif INPUT_SIZE == 18:
            # [0:8] is my stone occupied
            # [9:17] is other stone occupied
            # return np.array(state)[:, 0]
            # state
            # print("STATE_CONVERTER", state)
            # MAKE 18 (-1, 1 only)
            board = [1 if x==1 else 0 for x in state]
            board.extend([1 if x==-1 else 0 for x in state])
            # print("board", board)
            return board

        elif INPUT_SIZE == 27:
            # [0:8] is no stone occupied
            # [9:17] is other stone occupied
            # [18:26] is my stone occupied
            board = [1 if x==0 else 0 for x in state]
            board.extend([1 if x==1 else 0 for x in state])
            board.extend([1 if x==-1 else 0 for x in state])
            # print("board", board)
            return board

        else:
            raise Error

    def _next_action(self, state, available_actions):
        ob = OptimalBoard(state)
        converted_actions = ob.convert_available_actions(available_actions)
        converted_state = self.state_converter(ob.get_optimal_board())
        ###
        if self.train_mode:
            if random.random() < self.egreedy:
                action = random.choice(converted_actions)
            else:
                action = self.dqn.predict_one(converted_state)
        else:
            action = self.dqn.predict_one(converted_state)

        if action not in converted_actions:
            # 여기에 뭐를 학습으로 넣을 지 고민
            # 아니면, predict_one에서 필터를 넣을 지 고민
            self.dqn.add_train_set(converted_state, action, -1, self.state_converter([-1]*9), True)
            action = random.choice(converted_actions)

        original_action = ob.convert_to_original_action(action)

        return original_action

    def _feedback(self, state, action, next_state, reward, done):
        ob = OptimalBoard(state)
        converted_action = ob.convert_available_action(action)
        converted_state = self.state_converter(ob.get_optimal_board())
        next_ob = OptimalBoard(next_state)
        converted_next_state = self.state_converter(next_ob.get_optimal_board())

        self.dqn.add_train_set(converted_state, converted_action, reward, converted_next_state, done)
        self.dqn.study()

