''' qlearning algorithm '''
import random
import pickle

# TODO: 질문1: 갈수 없는 것도 값을 넣어야 하나?
# TODO: too many lint fix
class QLearning:
    ''' qlearning class '''
    def __init__(self, value, n_actions, learning_rate, discount, exploit_rate):
        self.init_value = value
        self.n_actions = n_actions
        self.learning_rate = learning_rate # alpha
        self.discount = discount # gamma
        self.exploit_rate = exploit_rate # epsilon
        self.q_scores = {}
        self.debug = False

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

    def get_available_scores(self, state, available_actions = None):
        self.init_state(state)
        if available_actions == None:
            available_actions = range(self.n_actions)
        scores = {i:self.q_scores[state][i] for i in available_actions}
        return scores

    def max(self, state, available_actions = None):
        scores = self.get_available_scores(state, available_actions)
        return max(scores.values())

    def argmax(self, state, available_actions = None):
        scores = self.get_available_scores(state, available_actions)
        return max(scores, key=scores.get)

    def rargmax(self, state, available_actions = None):
        scores = self.get_available_scores(state, available_actions)
        if self.debug: print("SCORES", scores)
        max_value = max(scores.values())
        if self.debug: print("MAX_VALUE", max_value)
        max_value_index = [k for k, v in scores.items() if v == max_value]
        if self.debug: print("MAX_VALUE_INDEX", max_value_index)
        if self.debug: print(scores, max_value, max_value_index, self.q_scores[state], available_actions)
        return random.choice(max_value_index)

    def rargmax_with_exploit(self, state, available_actions = None):
        if random.random() < self.exploit_rate:
            # print(len(available_actions),end='')
            if available_actions:
                return random.choice(available_actions)
            else:
                return random.randint(0, self.n_actions-1)
        else:
            return self.rargmax(state, available_actions)

    def set(self, state, action, value):
        self.init_state(state)
        self.q_scores[state][action] = value

    def learn(self, state, action, reward, next_state):
        q = self.get(state, action)
        dq = self.learning_rate*(reward + self.discount*self.max(next_state) - q)
        # print(state, action, next_state, reward, q, dq)
        self.q_scores[state][action] = q + dq
