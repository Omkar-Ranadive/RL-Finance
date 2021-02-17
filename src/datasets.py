import numpy as np
from torch.utils.data import IterableDataset
from torch.utils.data import Dataset
from collections import namedtuple


class ReplayBuffer(Dataset):
    def __init__(self, max_items):
        self.max_items = max_items
        self.items = []
        self.ptr = 0
        self.last_ptr = -1
        self.full_buffer = False
        self.Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))

    def append(self, *item):
        """
        Override the default append function to work as circular buffer
        Args:
            item (object): Transition (state, action, next_state, reward)
        Returns: Nothing (Item is appended to the circular buffer in place)
        """

        if self.full_buffer:
            self.items[self.ptr] = self.Transition(*item)
            self.last_ptr = self.ptr
            self.ptr = (self.ptr + 1) % self.max_items
        else:
            self.items.append(self.Transition(*item))
            self.last_ptr += 1
            if len(self.items) == self.max_items:
                self.ptr = 0
                self.last_ptr = 0
                self.full_buffer = True

    def random_sample(self, bs):
        indices = np.random.choice(len(self.items), bs - 1, replace=False)
        indices = np.append(indices, self.last_ptr)  # Always include the latest state pair
        return indices

    def __getitem__(self, index):
        return self.items[index]

    def __len__(self):
        return len(self.items)


class RLDataset(IterableDataset):
    """
    Dataset which gets updated as buffer gets filled
    """
    def __init__(self, buffer, sample_size):
        self.buffer = buffer
        self.sample_size = sample_size

    def __iter__(self):
        batch_indices = self.buffer.random_sample(self.sample_size)
        for index in batch_indices:
            yield self.buffer[index]
