from typing import Optional, List, Tuple
from functools import partial

import torch
import math
import random
from torch.utils.data.dataset import Dataset
from datasets import load_from_disk


MAX_LENGTH = 640


# я изменил сигнатуру, потому что строки нам тут нафиг не нужны
def collate_fn(
    batch: List[torch.Tensor], max_length: Optional[int] = MAX_LENGTH
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pad each sequence of the incoming sequences list
    :param batch: a list of the objects received from the dataset by __getitem__
    :param max_length: maximum sequence length to pad to (for "Brain" approach only)
    :return: tuple of padded sequences and corresponding training targets
    """
    
    max_length = max_length or max([len(obj) for obj in batch])
    padded = torch.vstack([
        torch.nn.functional.pad(obj, (0, max_length + 1 - len(obj)), value=0)
        for obj in batch
    ])
    # тут мы в качестве <eos> используем <pad>
    return padded[:, :-1], padded[:, 1:]



class WikiTextDataset(Dataset):
    def __init__(self, data_path, max_length):
        self.indices = load_from_disk(data_path)["indices"][:]
        self.max_length = max_length
        
    def __getitem__(self, index) -> torch.Tensor:
        return self.indices[index]

    def __len__(self):
        return len(self.indices)

    def get_collator(self):
        return partial(collate_fn, max_length=None)

    def get_batch_sampler(self):
        return None


class BrainDataset(WikiTextDataset):
    def get_collator(self):
        return partial(collate_fn, max_length=self.max_length)


# по факту такой датасет повторяет функциональность базового класса
class BigBrainDataset(WikiTextDataset):
    pass


# алгоритм следующий:
# датасет равномерно разбивается на нужное количество бинов
# в каждом бине индексы перемешиваются
# на каждой итерации случайно выбирается бин и оттуда достаются batch_size следующих индексов
# если в бине перестало хватать объектов на батч, оставшиеся добавляются в отдельный список rogues
# когда бины заканчиваются, индексы достаются уже втупую из rogues
class LengthSampler(torch.utils.data.Sampler):
    def __init__(self, dataset: WikiTextDataset, num_bins: int, batch_size: int):
        sorted_lens = list(sorted([(len(obj), idx) for idx, obj in enumerate(dataset)]))
        bin_size = int(math.ceil(len(dataset) / num_bins))
        self.batch_size = batch_size
        self.num_batches = len(dataset) // batch_size
        self.bins = [
            [obj[1] for obj in sorted_lens[i * bin_size: (i + 1) * bin_size]]
            for i in range(num_bins)
        ]
        self.num_bins = len(self.bins)
        
    def __len__(self):
        return self.num_batches

    def __iter__(self):
        for bin in self.bins:
            random.shuffle(bin)

        bins_shifts = [0] * self.num_bins
        available_bins = list(range(self.num_bins))
        rogues = []

        for bin_number in range(self.num_bins):
            if len(self.bins[bin_number]) < self.batch_size:
                rogues += self.bins[bin_number]
                available_bins.remove(bin_number)

        while len(available_bins) > 0:
            bin_number = random.choice(available_bins)
            batch = self.bins[bin_number][bins_shifts[bin_number]: bins_shifts[bin_number] + self.batch_size]
            bins_shifts[bin_number] += self.batch_size
            if bins_shifts[bin_number] + self.batch_size > len(self.bins[bin_number]):
                rogues += self.bins[bin_number][bins_shifts[bin_number]:]
                available_bins.remove(bin_number)
            yield batch

        i = 0
        while i + self.batch_size <= len(rogues):
            yield rogues[i: i + self.batch_size]
            i += self.batch_size


class UltraDuperBigBrainDataset(WikiTextDataset):
    def __init__(self, data_path, max_length, num_bins, batch_size):
        super().__init__(data_path, max_length)
        self.num_bins = num_bins
        self.batch_size = batch_size

    def get_batch_sampler(self):
        return LengthSampler(self, self.num_bins, self.batch_size)
