import os
import numpy as np
from typing import Optional

from dataset import AtariBase


class AtariTemporalDifference(AtariBase):
    """
    Atari dataset loader for temporal difference learning, i.e. returns
    current observation, action, reward, next observation, and done flag.
    """
    def __init__(
            self,
            path: str,
            part: Optional[int] = 1,
            subset: Optional[str] = 'all',
            buffer_len: Optional[int] = int(1e6),
            load_n_obs: Optional[bool] = True,
            load_n_acs: Optional[bool] = False
    ) -> None:
        """
        Loads Atari game data for temporal difference learning.

        Note: We assume that the n_acs are only loaded along with the n_obs.
        i.e. we do not handle the case where load_n_obs is False and
        load_n_acs is True. load_n_obs = False sets load_n_acs to False.

        :param path:
            Path to the dataset directory.
        :param part:
            Part of the dataset to load (e.g., 1, 2, etc.).
        :param subset:
            Subset of the data to load ('initial', 'final', or 'all').
        :param buffer_len:
            Length of the buffer for observations.
            Buffer size is 1 million in rlu_atari.
        :param load_n_obs:
            Whether to load next observation (n_obs) or not.
        :param load_n_acs:
            Whether to load next action (n_ac) or not.
        """
        super().__init__(path, part, subset, buffer_len)
        self.load_n_obs = load_n_obs
        self.load_n_acs = load_n_obs and load_n_acs
        self.valid_idxs = self.__get_valid_idxs(subset)

    def __get_valid_idxs(self, subset: str) -> np.ndarray:
        """
        Get valid indices for the specified subset.
        :param subset:
            Subset of the data to get valid indices for ('initial', 'final').
        :return:
            Array of valid indices.
        """
        valid_idxs_path = f"{self.path}/{self.part}/valid_idxs_{subset}.npy"
        if os.path.exists(valid_idxs_path):
            return np.load(valid_idxs_path)
        valid_idxs, idx, ptr = [], 0, 0
        for i in range(len(self.obs)):
            for j in range(self.buffer_len):
                if self.dones[ptr] == idx - 1:
                    valid_idxs = valid_idxs[:-3]
                    ptr += 1
                valid_idxs.append(idx)
                idx += 1
        valid_idxs = np.array(valid_idxs[:-3])
        np.save(valid_idxs_path, valid_idxs)
        return valid_idxs

    def __len__(self) -> int:
        """
        Get the number of sample-able data points in the dataset.
        :return:
            Number of valid indices in the dataset.
        """
        return len(self.valid_idxs)

    def __getitem__(self, i: int) -> tuple:
        """
        Get the ith sample from the dataset.
        :param i:
            Index of the sample to retrieve.
        :return:
            A tuple containing the current observation, action, reward,
            next observation (if load_n_obs is True), and done flag.
        """
        idx = self.valid_idxs[i]
        ob = np.zeros((4, 84, 84), dtype=np.uint8)
        for i in range(4):
            file_idx = (idx + i) // self.buffer_len
            sample_idx = (idx + i) % self.buffer_len
            ob[i] = self.obs[file_idx][sample_idx]
        ac = self.acs[idx]
        rew = self.rews[idx]
        done = idx == self.dones[np.searchsorted(self.dones, idx)]
        if not self.load_n_obs:
            return ob, ac, rew, done
        n_ob = np.zeros_like(ob)
        n_ob[:-1] = ob[1:]
        file_idx = min((idx + 4) // self.buffer_len, len(self.obs) - 1)
        sample_idx = (idx + 4) % self.buffer_len
        n_ob[-1] = self.obs[file_idx][sample_idx]
        if self.load_n_acs:
            n_ac = self.acs[min(idx + 4, len(self.acs) - 1)]
            return ob, ac, rew, n_ob, n_ac, done
        return ob, ac, rew, n_ob, done


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    dataset = AtariTemporalDifference(
        path="../../../rlu_atari/Breakout", part=1, subset='final', load_n_obs=False)
    dataloader = DataLoader(
        dataset, batch_size=2048, shuffle=False, num_workers=10)

    for i, batch in enumerate(dataloader):
        obs, acs, rews, dones = batch
        print(i, obs.shape, acs.shape, rews.shape, dones.shape)
