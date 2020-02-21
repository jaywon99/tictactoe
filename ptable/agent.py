''' prediction table agent '''

import random

import tictactoe.agent as agent
from tictactoe.utils import OptimalBoard

from .ptable import PredictionTable

class MyAgent(agent.AbstractAgent):
    def __init__(self, random_rate=0.2, learning_rate=0.1, debug=False):
        super().__init__()
        self.p_table = PredictionTable(learning_rate=learning_rate)
        self.random_rate = random_rate
        self.debug = debug

    def save(self, filename):
        self.p_table.save(filename)

    def load(self, filename):
        self.p_table.load(filename)

    def _next_action(self, state, actions):
        if self.train_mode and random.random() < self.random_rate:
            next_pos = random.choice(actions)
            if self.debug: print("SELECT", actions, "RANDOM", next_pos)

            return next_pos

        found_p = -1.0
        found_c = []
        
        ob = OptimalBoard(state)
        _id = ob.get_board_id()
        if self.debug: print("FROM", _id)

        scores = self.p_table.lookup(_id)
        converted_actions = ob.convert_available_actions(actions)
        for action in converted_actions:
            p = scores[action]
            if self.debug: print("ACTION", ob.convert_to_original_action(action), p)
            if p > found_p:
                found_p = p
                found_c = [ob.convert_to_original_action(action)]
            elif p == found_p:
                found_c.append(ob.convert_to_original_action(action))

        next_pos = random.choice(found_c)
        if self.debug: print("SELECT", found_c, found_p, next_pos)

        return next_pos

    def _episode_feedback(self, reward):
        # for winner
        (state, action, next_state, reward, done) = self.pop_history()
        np = reward
        ob = OptimalBoard(state)
        self.p_table.set(ob.get_board_id(), ob.convert_available_action(action), np)
        # print("RESULT", board_winner, t.get_board_id(board_winner), np)
        while len(self.history) > 0:
            (state, action, next_state, reward, done) = self.pop_history()
            ob = OptimalBoard(state)
            np = self.p_table.learn(ob.get_board_id(), ob.convert_available_action(action), np)
            # print("LEARN", board_winner, t.get_board_id(board_winner), np)

