import os
import copy
import torch
import numpy as np
import torch.optim as optim

from policies import BasePolicy


class BaseAgent:
    def __init__(
        self, policy: BasePolicy, lr: float,
        weight_decay: float, model_path: str, save_optimizer: bool,
        resume_training: bool, device: torch.device
    ):
        self.device = device
        self.model_path = model_path
        self.optimizer_path = os.path.join(model_path, "optimizer.pt")
        self.target_path = os.path.join(model_path, "target_model.pt")
        self.model_path = os.path.join(model_path, "model.pt")
        self.save_optimizer = save_optimizer
        self.resume_training = resume_training
        self.policy = policy
        self.target_policy = copy.deepcopy(policy)
        self.optimizer = optim.Adam(
            self.policy.model.parameters(), lr=lr, weight_decay=weight_decay)  # type: ignore
        if self.resume_training:
            self.load()
        self.policy.train()
        self.target_policy.eval()

    def act(self, obs: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Subclass should override this method")

    def learn(
            self, obs: np.ndarray, actions: np.ndarray, rewards: np.ndarray,
            next_obs: np.ndarray, next_actions: np.ndarray, dones: np.ndarray
    ):
        raise NotImplementedError("Subclass should override this method")

    def save(self):
        torch.save(self.policy.model.state_dict(), self.model_path)  # type: ignore
        torch.save(self.target_policy.model.state_dict(), self.target_path)  # type: ignore
        if self.save_optimizer:
            torch.save(self.optimizer.state_dict(), self.optimizer_path)

    def load(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file {self.model_path} does not exist.")
        self.policy.model.load_state_dict(torch.load(self.model_path, map_location=self.device))  # type: ignore
        if not os.path.exists(self.target_path):
            raise FileNotFoundError(f"Target model file {self.target_path} does not exist.")
        self.target_policy.model.load_state_dict(torch.load(self.target_path, map_location=self.device))  # type: ignore
        if self.save_optimizer and os.path.exists(self.optimizer_path):
            self.optimizer.load_state_dict(torch.load(self.optimizer_path))
        else:
            raise FileNotFoundError(f"Optimizer file {self.optimizer_path} does not exist.")
