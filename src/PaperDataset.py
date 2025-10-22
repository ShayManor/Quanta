from queue import Queue
from typing import Optional

from datasets import load_dataset
from torch.utils.data import Dataset


class PaperDataset(Dataset):
    def __init__(self, source: str, length: int = -1, split: Optional[str] = None, cache_size: int = 20):
        super().__init__()
        self.length = length
        self.cache = Queue(maxsize=cache_size)
        self.cache_size = cache_size
        self.cache_total_size = cache_size

        if self.length != -1:
            if split:
                self.stream_dataset = load_dataset(
                    source, streaming=True, split=split
                )
            else:
                self.stream_dataset = load_dataset(
                    source, streaming=True
                )
        self.cache.put(self.stream_dataset.take(cache_size).to_list())

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        ret = self.cache.get()
        self.cache_size -= 1

        if self.cache_size < self.cache_total_size // 2:
            self.cache.put(self.stream_dataset.take(self.cache_total_size // 2))

        return ret
