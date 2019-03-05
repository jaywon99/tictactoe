import random

from agent import Agent

class tictactoe_converter:
    CONVS = ['123456789',
             '369258147',
             '987654321',
             '741852963',
             '321654987',
             '963852741',
             '789456123',
             '147258369']

    def __init__(self):
        pass

    def find_mapper(self, state):
        s = state
        board = [0] * 9
        for i in range(9):
            board[i] = s & 0x03
            s = s >> 2
        # print(board)

        _id = 4 << 18
        mapping_id = -1
        for i in range(len(tictactoe_converter.CONVS)):
            alternate = [board[int(j)-1] for j in tictactoe_converter.CONVS[i]]
            _alt_id = 0
            for digit in alternate:
                _alt_id = (_alt_id << 2) | digit
            if _alt_id < _id:
                _id = _alt_id
                mapping_id = i
        return mapping_id

    def get_optimized_state(self, state, _id):
        s = state
        board = [0] * 9
        for i in range(9):
            board[i] = s & 0x03
            s = s >> 2
        # print(board)

        alternate = [board[int(j)-1] for j in tictactoe_converter.CONVS[_id]]
        _alt_id = 0
        for digit in alternate:
            _alt_id = (_alt_id << 2) | digit
        return _alt_id

    def map(self, action, _id):
        return int(tictactoe_converter.CONVS[_id][action])-1

    def invert(self, action, _id):
        return tictactoe_converter.CONVS[_id].find(str(action+1))

'''
c = tictactoe_converter()
print(c.find_mapper(1 | (2<<2)))
print(c.find_mapper(1<<4 | (2<<2)))
print(c.find_mapper(1<<(2*6) | 2<<(2*7)))
print(c.find_mapper(1<<(2*8) | 2<<(2*7)))

print(c.get_optimized_state(1 | (2<<2), 2))
print(c.get_optimized_state(1<<4 | (2<<2), 6))
print(c.get_optimized_state(1<<(2*6) | 2<<(2*7), 4))
print(c.get_optimized_state(1<<(2*8) | 2<<(2*7), 0))

print(c.map(2, 2))
print(c.map(0, 6))
print(c.map(8, 4))
print(c.map(6, 0))

print(c.invert(6, 2))
print(c.invert(6, 6))
print(c.invert(6, 4))
print(c.invert(6, 0))
'''


class QAgent(Agent):
    def __init__(self, name, qlearning):
        super().__init__(name)
        self.q = qlearning
        self.mode = 0
        self.c = tictactoe_converter()

    def set_mode(self, mode):
        '''mode=1: learning
        mode=0: real
        '''
        self.mode = mode

    def _next_action(self, state, possible_actions):
        _mapper = self.c.find_mapper(state)
        _state = self.c.get_optimized_state(state, _mapper)
        _possible_actions = [self.c.map(v, _mapper) for v in possible_actions]
        if self.mode == 1:
            action = self.q.rargmax_with_exploit(_state, _possible_actions)
        else:
            action = self.q.rargmax(_state, _possible_actions)
        self.last_state = _state
        self.last_action = action
        return self.c.invert(action, _mapper)

    def next_action(self, state, possible_actions):
        if self.mode == 1:
            action = self.q.rargmax_with_exploit(state, possible_actions)
        else:
            action = self.q.rargmax(state, possible_actions)
        self.last_state = state
        self.last_action = action
        return action

    def _learn(self, next_state, reward):
        _mapper = self.c.find_mapper(next_state)
        _next_state = self.c.get_optimized_state(next_state, _mapper)
        self.q.learn(self.last_state, self.last_action, reward, _next_state)

    def learn(self, next_state, reward):
        self.q.learn(self.last_state, self.last_action, reward, next_state)

