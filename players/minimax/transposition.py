import pickle

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

    def serialize(self):
        return pickle.dumps(self.table)

    def deserialize(self, obj):
        if obj != None:
            self.table = pickle.loads(obj)

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

    def serialize(self):
        return pickle.dumps(self.table)

    def deserialize(self, obj):
        if obj != None:
            self.table = pickle.loads(obj)
