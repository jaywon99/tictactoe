''' DQN learning '''

import random

import tictactoe.agent as agent
from tictactoe.utils import OptimalBoard

from .dqn import DQN

INPUT_SIZE = 18
class MyAgent(agent.AbstractAgent):
    ''' DQN Learning Agent '''
    def __init__(self, egreedy=0.2):
        super().__init__()
        self.mode = 0
        self.network = DQN(layers=[INPUT_SIZE, 54, 54, 9], lr=0.001)
        self.egreedy = egreedy

    def save(self, filename):
        self.network.save(filename)

    def load(self, filename):
        self.network.load(filename)

    def set_session(self, session):
        ''' set tensorflow session '''
        self.network.set_session(session)

    @staticmethod
    def convert_state(state):
        ''' convert state from board style to NN style '''

        if INPUT_SIZE == 9:
            # MAKE 9 / (-1, 0, 1)로 되어있는 board state
            return state

        if INPUT_SIZE == 18:
            # [0:8] is my stone occupied
            # [9:17] is other stone occupied
            board = [1 if x == 1 else 0 for x in state]
            board.extend([1 if x == -1 else 0 for x in state])
            return board

        if INPUT_SIZE == 27:
            # [0:8] is no stone occupied
            # [9:17] is other stone occupied
            # [18:26] is my stone occupied
            board = [1 if x == 0 else 0 for x in state]
            board.extend([1 if x == 1 else 0 for x in state])
            board.extend([1 if x == -1 else 0 for x in state])
            return board

        raise Exception

    def _next_action(self, state, available_actions):
        optimal_board = OptimalBoard(state)
        converted_actions = optimal_board.convert_available_actions(available_actions)
        converted_state = self.convert_state(optimal_board.get_optimal_board())
        ###
        if self.train_mode:
            if random.random() < self.egreedy:
                action = random.choice(converted_actions)
            else:
                action = self.network.predict_one(converted_state)
        else:
            action = self.network.predict_one(converted_state)

        if action not in converted_actions:
            # 여기에 뭐를 학습으로 넣을 지 고민
            # 아니면, predict_one에서 필터를 넣을 지 고민
            self.network.add_train_set(converted_state, action, -1, self.convert_state([-1]*9), True)
            action = random.choice(converted_actions)

        original_action = optimal_board.convert_to_original_action(action)

        return original_action

    def _feedback(self, state, action, next_state, reward, done):
        state_ob = OptimalBoard(state)
        converted_action = state_ob.convert_available_action(action)
        converted_state = self.convert_state(state_ob.get_optimal_board())
        next_ob = OptimalBoard(next_state)
        converted_next_state = self.convert_state(next_ob.get_optimal_board())

        self.network.add_train_set(converted_state,
                                   converted_action,
                                   reward,
                                   converted_next_state,
                                   done)
        self.network.study()
