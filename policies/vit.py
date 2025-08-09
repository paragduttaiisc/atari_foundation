import torch
from policies import BasePolicy


class ViTPolicy(BasePolicy):
    def __init__(self) -> None:
        super().__init__()
        self.name = 'ViT'
        self.model = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
