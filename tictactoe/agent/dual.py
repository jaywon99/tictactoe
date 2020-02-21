class DualAgent:
    def __init__(self, agent1, agent2):
        self.agent1 = agent1
        self.agent2 = agent2
        self.current = None

    def reset(self):
        self.current = None
        self.agent1.reset()
        self.agent2.reset()

    def set_train_mode(self, mode):
        self.agent1.set_train_mode(mode)
        self.agent2.set_train_mode(mode)

    def next_agent(self):
        if self.current:
            self.current = self.other_agent()
        else:
            self.current = self.agent1
        return self.current

    def current_agent(self):
        return self.current

    def other_agent(self):
        return self.agent1 if self.agent2 == self.current else self.agent2

