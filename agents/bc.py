import torch
import numpy as np
import torch.nn.functional as F

from agents import BaseAgent


class BCAgent(BaseAgent):
    """
    BC Agent class that inherits from BaseAgent. Implements the Behavior
    Cloning (BC) algorithm on offline data for policy extraction.
    """
    def __init__(self, ema: float, **kwargs):
        super().__init__(**kwargs)
        self.ema = ema

    def act(self, obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            action = self.target_policy(obs)  # type: ignore
        return action

    def learn(
            self,
            obs: np.ndarray,
            acs: np.ndarray,
            rews: np.ndarray = np.zeros((1,)),
            n_obs: np.ndarray = np.zeros((1,)),
            n_acs: np.ndarray = np.zeros((1,)),
            dones: np.ndarray = np.zeros((1,))
    ) -> float:
        actions_t = torch.tensor(acs, dtype=torch.long, device=self.device)
        # Forward pass
        logits = self.policy.model(obs)  # type: ignore
        log_probs = F.log_softmax(logits, dim=-1)
        # Compute loss
        loss = F.nll_loss(log_probs, actions_t)
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # update target
        for param, target_param in zip(
            self.policy.model.parameters(),  # type: ignore
            self.target_policy.model.parameters()  # type: ignore
        ):
            target_param.data.copy_(
                self.ema * target_param.data + (1 - self.ema) * param.data)
        # return
        return loss.item()
