import numpy as np
from typing import Optional

from dataset import AtariBase


class AtariMonteCarlo(AtariBase):
    def __init__(
            self,
            path: str,
            part: Optional[int] = 1,
            subset: Optional[str] = 'all',
            buffer_len: Optional[int] = int(1e6),
            trajectory_max_len: Optional[int] = 128
    ) -> None:
        """
        Loads Atari game data for Monte Carlo learning.
        :param path:
            Path to the dataset directory.
        :param part:
            Part of the dataset to load (e.g., 1, 2, etc.).
        :param subset:
            Subset of the data to load ('initial', 'final', or 'all').
        :param buffer_len:
            Length of the buffer for observations.
            Buffer size is 1 million in rlu_atari.
        :param trajectory_max_len:
            Maximum length of a trajectory.
        """
        super().__init__(path, part, subset, buffer_len)
        self.trajectory_max_len = trajectory_max_len

    def __len__(self) -> int:
        """
        Get the number of trajectories in the dataset.
        :return:
            Number of done flags in the dataset.
        """
        return len(self.dones)

    def __getitem__(self, idx: int) -> tuple:
        """
        Get the ith trajectory from the dataset.
        :param idx:
            Index of the trajectory to retrieve.
        :return:
            A tuple containing the sequence of observations, actions,
            and rewards for the trajectory.
            The trajectory is padded with zeros if it is shorter than
            trajectory_max_len and the valid sequence length is also returned.
        """
        start_idx = (self.dones[idx - 1] + 1) if idx else 0
        end_idx = self.dones[idx]
        traj_len = end_idx - start_idx + 1
        if traj_len > self.trajectory_max_len:
            rand_idx = np.random.randint(0, traj_len - self.trajectory_max_len)
            traj_len = self.trajectory_max_len
            start_idx += rand_idx
            end_idx = start_idx + traj_len - 1
        assert start_idx < end_idx
        obs = np.zeros((self.trajectory_max_len, 4, 84, 84), dtype=np.uint8)
        acs = np.zeros(self.trajectory_max_len, dtype=np.int32)
        rews = np.zeros(self.trajectory_max_len, dtype=np.float32)
        for i in range(traj_len):
            file_idx = (start_idx + i) // self.buffer_len
            sample_idx = (start_idx + i) % self.buffer_len
            obs[i] = self.obs[file_idx][sample_idx]
            acs[i] = self.acs[start_idx + i]
            rews[i] = self.rews[start_idx + i]
        return obs, acs, rews, traj_len


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    dataset = AtariMonteCarlo(
        path="../../../rlu_atari/Breakout", part=1, trajectory_max_len=128)
    dataloader = DataLoader(
        dataset, batch_size=32, shuffle=False, num_workers=8)

    for i, batch in enumerate(dataloader):
        obs, acs, rews, traj_len = batch
        print(i, obs.shape, acs.shape, rews.shape, traj_len.shape)
