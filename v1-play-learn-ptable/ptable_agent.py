import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "lib"))

import tictactoe.agent as agent
from tictactoe.utils import compact_observation 

import random
from ptable import PredictionTable 

class SmartAgent(agent.AbstractAgent):
    def __init__(self, p_table, random_rate=0.2, debug=False):
        super().__init__()
        self.p_table = p_table
        self.random_rate = random_rate
        self.debug = debug

    def _history_compaction(self, obs, action):
        return (compact_observation(obs), action)

    def _next_action(self, obs, actions):
        if random.random() < self.random_rate:
            next_pos = random.choice(actions)
            if self.debug: print("SELECT", actions, "RANDOM", next_pos)

            return next_pos

        found_p = -1.0
        found_c = []
        
        _id = compact_observation(obs)
        if self.debug: print("FROM", _id)

        scores = self.p_table.lookup(_id)
        for action in actions:
            p = scores[action]
            if self.debug: print("ACTION", action, p)
            if p > found_p:
                found_p = p
                found_c = [action]
            elif p == found_p:
                found_c.append(action)

        next_pos = random.choice(found_c)
        if self.debug: print("SELECT", found_c, found_p, next_pos)

        return next_pos

    def feedback(self, reward):
        # for winner
        (state, action) = self.pop_history()
        np = reward
        self.p_table.set(state, action, np)
        # print("RESULT", board_winner, t.get_board_id(board_winner), np)
        while len(self.history) > 0:
            (state, action) = self.pop_history()
            np = self.p_table.learn(state, action, np)
            # print("LEARN", board_winner, t.get_board_id(board_winner), np)

