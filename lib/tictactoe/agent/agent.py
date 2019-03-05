import random

class AbstractAgent:
    def __init__(self):
        self.history = []

    def _reset(self):
        pass

    def reset(self):
        self.history = []
        self._reset()

    def _history_compaction(self, obs, action):
        return (obs, action)

    def save_history(self, obs, action):
        self.history.append(self._history_compaction(obs, action))

    def pop_history(self):
        return self.history.pop()

    def all_history(self):
        return self.history

    def next_action(self, obs, availables_actions):
        action = self._next_action(obs, availables_actions)
        self.save_history(obs, action)
        return action

    def _next_action(self, obs, availables_actions):
        raise NotImplementedError

class RandomAgent(AbstractAgent):
    def __init__(self):
        super().__init__()

    def _next_action(self, obs, availables_actions):
        return random.choice(availables_actions)


