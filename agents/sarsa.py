from agents import BaseAgent


class SarsaAgent(BaseAgent):
    """
    SARSA Agent class that inherits from BaseAgent. Implements the SARSA
    (State-Action-Reward-State-Action) algorithm on offline data for policy
    extraction.
    """
    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)
