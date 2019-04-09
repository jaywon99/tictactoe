import random

class AbstractAgent:
    def __init__(self):
        self.history = []
        self.train_mode = False

    def _reset(self, feedback, episode):
        pass

    def reset(self, feedback, episode = -1):
        self.train_mode = feedback
        self.history = []
        self._reset(feedback, episode)

    def _history_compaction(self, state, action):
        return (state, action)

    def save_history(self, state, action):
        self.history.append(self._history_compaction(state, action))

    def pop_history(self):
        return self.history.pop()

    def all_history(self):
        return self.history

    def next_action(self, state, availables_actions):
        action = self._next_action(state, availables_actions)
        self.save_history(state, action)
        return action

    def _next_action(self, state, availables_actions):
        raise NotImplementedError

    def feedback(self, next_state, reward, done):
        pass

    def episode_feedback(self, reward):
        pass

    def save(self, path):
        pass

    def load(self, path):
        pass

class RandomAgent(AbstractAgent):
    def __init__(self):
        super().__init__()

    def _next_action(self, state, availables_actions):
        return random.choice(availables_actions)


