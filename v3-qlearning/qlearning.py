import random
import pickle

# 질문1: 갈수 없는 것도 값을 넣어야 하나?
class qlearning:
    def __init__(self, value, n_actions, learning_rate, discount, exploit_rate):
        self.init_value = value
        self.n_actions = n_actions
        self.learning_rate = learning_rate # alpha
        self.discount = discount # gamma
        self.exploit_rate = exploit_rate # epsilon
        self.q_scores = {}

    def load(self, filename):
        try:
            with open(filename, 'rb') as f:
                (self.q_scores) = pickle.load(f)
        except:
            self.q_scores = {}

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump((self.q_scores), f)

    def init_state(self, state):
        if state not in self.q_scores:
            self.q_scores[state] = [self.init_value] * self.n_actions

    def get(self, state, action):
        self.init_state(state)
        return self.q_scores[state][action]

    def get_possible_scores(self, state, possible_actions = None):
        self.init_state(state)
        if possible_actions == None:
            possible_actions = range(self.n_actions)
        scores = {i:self.q_scores[state][i] for i in possible_actions}
        return scores

    def max(self, state, possible_actions = None):
        scores = self.get_possible_scores(state, possible_actions)
        return max(scores.values())

    def argmax(self, state, possible_actions = None):
        scores = self.get_possible_scores(state, possible_actions)
        return max(scores, key=scores.get)

    def rargmax(self, state, possible_actions = None):
        scores = self.get_possible_scores(state, possible_actions)
        # print("SCORES", scores)
        max_value = max(scores.values())
        # print("MAX_VALUE", max_value)
        max_value_index = [k for k, v in scores.items() if v == max_value]
        # print("MAX_VALUE_INDEX", max_value_index)
        # print(scores, max_value, max_value_index, self.q_scores[state], possible_actions)
        return random.choice(max_value_index)

    def rargmax_with_exploit(self, state, possible_actions = None):
        if random.random() < self.exploit_rate:
            # print(len(possible_actions),end='')
            if possible_actions:
                return random.choice(possible_actions)
            else:
                return random.randint(0, self.n_actions-1)
        else:
            return self.rargmax(state, possible_actions)

    def set(self, state, action, value):
        self.init_state(state)
        self.q_scores[state][action] = value

    def learn(self, state, action, reward, next_state):
        q = self.get(state, action)
        dq = self.learning_rate*(reward + self.discount*self.max(next_state) - q)
        # print(state, action, next_state, reward, q, dq)
        self.q_scores[state][action] = q + dq

