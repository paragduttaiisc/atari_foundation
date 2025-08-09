import torch.optim as optim


class BaseAgent:
    def __init__(self, env, policy=None):
        self.env = env
        self.policy = policy
        self.optimizer = optim.Adam(self.policy.model.parameters(), lr=0.001)

    def act(self):
        raise NotImplementedError("This method should be overridden by subclasses.")

    def learn(self, reward, next_state):
        raise NotImplementedError("This method should be overridden by subclasses.")
