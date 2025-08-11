import gym
import torch
import numpy as np
from typing import Tuple
from torch.utils.data import DataLoader

from agents import BaseAgent
from dataset import AtariBase


def simulate(
        env: gym.Env, agent: BaseAgent, eps: float = 0.00065536,
        max_steps: int = 27000
) -> Tuple[float, int]:
    # TODO: handle ale_lives
    obs, _ = env.reset()
    buffer = [np.zeros_like(obs) for _ in range(3)] + [obs]
    done = False
    total_reward = 0
    eval_steps = 0
    while not done:
        if eval_steps >= max_steps:
            break
        if np.random.rand() <= eps:
            action = env.action_space.sample()
        else:
            action = agent.act(np.array(buffer))[0].argmax().item()
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        buffer = buffer[1:] + [obs]
        eval_steps += 1
    return total_reward, eval_steps


class OfflineRLTrainer:
    def __init__(
            self,
            env: gym.Env,
            agent: BaseAgent,
            dataset: AtariBase,
            batch_size: int,
            num_steps: int,
            eval_freq: int,
            device: torch.device
    ) -> None:
        self.env = env
        self.agent = agent
        self.num_steps = num_steps + 1
        self.device = device

        step = 0
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        for batch in dataloader:
            if step > self.num_steps:
                break

            if step % eval_freq == 0:
                total_eval_steps = 0
                reward_list = []
                for _ in range(10):
                    if 125_000 - total_eval_steps <= 0:
                        break
                    reward, eval_steps = simulate(env, agent)
                    reward_list.append(reward)
                    total_eval_steps += eval_steps
                avg_reward = np.mean(reward_list)
                print(f"Evaluation results after {step} steps:")
                print(f"Average reward: {avg_reward:.2f}")
            
            batch = {k: v.to(self.device) for k, v in batch.items()}
            self.agent.learn(batch)
            step += 1