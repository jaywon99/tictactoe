import random

class AbstractAgent:
    def __init__(self):
        self.history = []
        self.train_mode = False

    def set_train_mode(self, train_mode):
        self.train_mode = train_mode

    def _reset(self):
        ''' reset agent status internal (real class implement this)
        '''
        pass

    def reset(self):
        ''' reset agent for general purpose.
        It will call _reset internally.
        '''
        self.history = []
        self._reset()

    def is_played(self):
        return len(self.history) > 0

    def save_history(self, state, action):
        ''' to replay, save all history
        '''
        self.history.append([state, action])

    def pop_history(self):
        ''' backward replay from play book
        '''
        return self.history.pop()

    def all_history(self):
        ''' get all history from 1st move
        '''
        return self.history

    def _next_action(self, state, availables_actions):
        ''' internal code for next_action
        '''
        raise NotImplementedError

    def next_action(self, state, availables_actions):
        ''' call _next_action and save play history
        '''
        action = self._next_action(state, availables_actions)
        self.save_history(state, action)
        return action

    def _feedback(self, state, action, next_state, reward, done):
        ''' feedback of each play - generally feedback came after counter-part playing
        need to implement this.
        '''
        pass

    def feedback(self, next_state, reward, done):
        ''' feedback of each play - generally feedback came after counter-part playing
        call _feedback internally if train_mode is on
        '''
        self.history[-1].extend([next_state, reward, done])
        if self.train_mode:
            self._feedback(self.history[-1][0], self.history[-1][1], self.history[-1][2], self.history[-1][3], self.history[-1][4])

    def _episode_feedback(self, reward):
        ''' feedback after full game finished.
        need to implement by process
        '''
        pass

    def episode_feedback(self, reward):
        ''' feedback after full game finished.
        call _episode_feedback internally if train_mode is on
        '''
        if self.train_mode:
            self._episode_feedback(reward)

    def save(self, path):
        ''' save intelligence to storage
        '''
        pass

    def load(self, path):
        ''' load intelligence from storage
        '''
        pass

class RandomAgent(AbstractAgent):
    def __init__(self):
        super().__init__()

    def _next_action(self, state, availables_actions):
        return random.choice(availables_actions)


