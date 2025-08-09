import torch.nn as nn

from policies import BasePolicy


class CNNPolicy(BasePolicy):
    def __init__(self, action_space: int) -> None:
        """
        Initialize the CNN policy.
        :param action_space:
            The number of actions in the action space.
        """
        super().__init__()
        self.name = 'CNN'
        self.model = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),  # Input: (B, 4, 84, 84)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),  # Output: (B, 64, 7, 7)
            nn.ReLU(),
            nn.Flatten(), # Output: (B, 3136)
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, action_space)  # Output: (B, action_space)
        )
