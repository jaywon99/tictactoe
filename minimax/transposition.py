
class TranspositionTable:
    def __init__(self):
        self.table = {}

    def get(self, key):
        if key in self.table:
            return self.table[key]
        else:
            return None

    def put(self, key, value):
        self.table[key] = value

class ABPTranspositionTable: # Alpha-Beta-Pruning Transposition Table
    LOWERBOUND, EXACT, UPPERBOUND = -1,0,1
    def __init__(self):
        self.table = {}

    def get(self, key):
        if key in self.table:
            return self.table[key]
        else:
            return None

    def put(self, key, depth, value, flag):
        self.table[key] = {'depth': depth, 'value': value, 'flag': flag}

