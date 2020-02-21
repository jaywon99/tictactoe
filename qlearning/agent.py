''' qlearning agent '''

import tictactoe.agent as agent
from tictactoe.utils import OptimalBoard

from .qlearning import QLearning

class MyAgent(agent.AbstractAgent):
    ''' qlearning agent '''
    def __init__(self, learning_rate=0.1, discount_rate=0.9, exploit_rate=0.2):
        super().__init__()
        self.q = QLearning(value=0, n_actions=9, learning_rate=learning_rate, discount=discount_rate, exploit_rate=exploit_rate)

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
        ob2 = OptimalBoard(next_state)
        self.q.learn(ob1.get_board_id(),
                     ob1.convert_available_action(action),
                     reward,
                     ob2.get_board_id())
