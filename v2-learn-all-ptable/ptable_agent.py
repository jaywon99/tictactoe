import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "lib"))

import tictactoe.agent as agent
from tictactoe.utils import compact_observation 

import random
from ptable import PredictionTable 

import collections

class SmartAgent(agent.AbstractAgent):
    def __init__(self, p_table, random_rate=0.2, debug=False):
        super().__init__()
        self.p_table = p_table
        self.random_rate = random_rate
        self.debug = debug
        self.q = collections.deque(maxlen=10)

    def _history_compaction(self, obs, action):
        return (compact_observation(obs), action)

    def set_queue(self, seq):
        self.q.extend(seq)

    def _next_action(self, obs, candidates):
        return self.q.popleft()

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

