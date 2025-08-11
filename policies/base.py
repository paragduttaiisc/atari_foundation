import torch
import numpy as np
import torch.nn as nn

from typing import Union


class BasePolicy(nn.Module):
    """
    Base class for all policies.
    """
    def __init__(self) -> None:
        super(BasePolicy, self).__init__()
        self.name = None
        self.model = None
        self.device = None

    def forward(
            self,
            x: Union[torch.Tensor, np.ndarray],
            requires_grad: bool = False
    ) -> torch.Tensor:
        assert self.model is not None
        # Check if the input is a tensor else convert it to a tensor
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        elif not isinstance(x, torch.Tensor):
            raise TypeError(
                f"Input must be a numpy array or a torch tensor, got {type(x)}")
        # reshape input if necessary depending on the model
        if self.name == 'MLP':
            if x.ndim > 2:
                raise ValueError(
                    "Input tensor must have at most 2 dimensions for MLP.")
            if x.ndim == 1:
                x = x.unsqueeze(0)
        elif self.name == 'CNN':
            if x.ndim < 3 or x.ndim > 4:
                raise ValueError(
                    "Input tensor must have exactly 3 or 4 dimensions for CNN.")
            if x.ndim == 3:
                x = x.unsqueeze(0)
        elif self.name == 'ViT':
            raise NotImplementedError(
                "ViT model is not implemented in this base policy.")
        else:
            raise ValueError(
                f"Invalid model type: {self.name}")
        # check model is initialized
        if self.model is None:
            raise ValueError("Model is not initialized. Please check the policy.")
        # send data to device
        if self.device is not None:
            x = x.to(self.device)
        # forward pass
        if requires_grad:
            return self.model(x)
        with torch.no_grad():
            return self.model(x)

    def __repr__(self) -> str:
        return f"AtariPolicy Variant={self.name}"
