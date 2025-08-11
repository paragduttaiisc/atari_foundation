import numpy as np
from torch.utils.data import Dataset
from typing import Optional


class AtariBase(Dataset):
    """
    Atari dataset loader base class.
    """
    def __init__(
            self,
            path: str,
            part: int = 1,
            subset: str = 'all',  # 'initial', 'final', or 'all'
            buffer_len: int = int(1e6),
    ) -> None:
        """
        Loads Atari game data from a specified path.
        :param path:
            Path to the dataset directory.
        :param part:
            Part of the dataset to load (e.g., 1, 2, etc.).
        :param subset:
            Subset of the data to load ('initial', 'final', or 'all').
        :param buffer_len:
            Length of the buffer for observations.
            Buffer size is 1 million in rlu_atari.
        """
        assert subset in ['initial', 'final', 'all']
        self.path = path
        self.part = part
        self.buffer_len = buffer_len
        self.obs = []
        self.acs = None
        self.rews = None
        self.dones = None
        self.__load_data(subset)

    def __load_data(self, subset: str) -> None:
        """
        Load data from the specified path and subset.
        :param subset:
            Subset of the data to load ('initial', 'final', or 'all').
        :return:
            None
        """
        if subset == 'initial':
            data_idxs = np.arange(0, 2)
        elif subset == 'final':
            data_idxs = np.arange(8, 10)
        else:  # 'all'
            data_idxs = np.arange(0, 10)
        for idx in data_idxs:
            file_path = f"{self.path}/{self.part}/obs_{idx}.npy"
            self.obs.append(np.load(file_path, mmap_mode='r'))
        start_idx = data_idxs[0] * self.buffer_len
        end_idx = (data_idxs[1] + 1) * self.buffer_len
        self.acs = np.load(f"{self.path}/{self.part}/acs.npy")
        self.acs = self.acs[start_idx:end_idx]
        self.rews = np.load(f"{self.path}/{self.part}/rews.npy")
        self.rews = self.rews[start_idx:end_idx]
        self.dones = np.load(f"{self.path}/{self.part}/dones.npy")
        start_idx2 = np.searchsorted(self.dones, start_idx)
        end_idx2 = np.searchsorted(self.dones, end_idx)
        self.dones = self.dones[start_idx2:end_idx2]
        if self.dones[-1] != (end_idx - 1):
            self.dones = np.append(self.dones, end_idx - 1)
        self.dones -= start_idx

    def __len__(self):
        raise NotImplementedError("This method should be overridden by subclasses.")

    def __getitem__(self, i: int):
        raise NotImplementedError("This method should be overridden by subclasses.")

    def __repr__(self) -> str:
        return f"AtariDataset (path={self.path}, seed={self.part}, samples={len(self)})"
