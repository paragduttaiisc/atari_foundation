import torch
import torch.nn as nn

from policies import BasePolicy


class MLPPolicy(BasePolicy):
    def __init__(
            self, state_space: int, action_space: int, hidden_dim: int = 128
    ) -> None:
        """
        Initialize the MLP policy.
        :param state_space:
            The size of the state space (number of features).
        :param action_space:
            The number of actions in the action space.
        :param hidden_dim:
            The number of hidden units in the MLP.
        """
        # TODO: Make dynamic number of hidden layers
        super().__init__()
        self.name = 'MLP'
        self.model = nn.Sequential(
            nn.Linear(state_space, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_space)
        )
