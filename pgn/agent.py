''' DQN learning '''

import random

import tictactoe.agent as agent
from tictactoe.utils import OptimalBoard

from .pgn import PGN

INPUT_SIZE = 18
GAMMA = 0.8
class MyAgent(agent.AbstractAgent):
    ''' DQN Learning Agent '''
    def __init__(self, egreedy=0.2):
        super().__init__()
        self.mode = 0
        self.network = PGN(layers=[INPUT_SIZE, 54, 54, 9], lr=0.001)
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

        # 여기에 무엇을 넣을 지 고민
        if action not in converted_actions:
            # 여기에 뭐를 학습으로 넣을 지 고민
            # 아니면, predict_one에서 필터를 넣을 지 고민
            self.network.add_train_set(converted_state,
                                       action,
                                       -1,
                                       self.convert_state([-1]*9),
                                       True)
            action = random.choice(converted_actions)

        original_action = optimal_board.convert_to_original_action(action)

        return original_action

    def _calculate_reward(self, history, final_reward):
        ''' convert turn history to learning data and
        calculate reward (multiply gamma)
        '''

        replay_buffer = []
        size = len(history)
        for idx, turn in enumerate(history):
            optimal_board = OptimalBoard(turn[self.HISTORY_STATE])
            converted_action = optimal_board.convert_available_action(turn[self.HISTORY_ACTION])
            converted_state = self.convert_state(optimal_board.get_optimal_board())
            replay_buffer.append([converted_state, 
                                  converted_action, 
                                  final_reward * GAMMA ** (size-idx-1)])

        running_add = final_reward
        for i in reversed(range(len(replay_buffer))):
            replay_buffer[i][2] = running_add   # 2 is reward of every turn
            running_add = running_add * GAMMA

        return replay_buffer

    def _episode_feedback(self, reward):
        replay_buffer = self._calculate_reward(self.history, (reward*0.5)+0.5)

        self.network.add_to_replay_buffer(reward, replay_buffer)
        self.network.study()
