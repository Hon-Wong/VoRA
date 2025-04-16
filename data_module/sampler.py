import random
from typing import List, Sized

from torch.utils.data import RandomSampler, Sampler


class GroupRandomSampler(RandomSampler):
    # Only shuffle inside a dataset
    def __init__(self, data_source: Sized, lengths: List[int]):
        super().__init__(data_source=data_source)
        self.lengths = lengths
        self.start_idx = [0]
        for length in self.lengths:
            self.start_idx.append(self.start_idx[-1] + length)
        self.samplers = [RandomSampler(list(range(length))) for length in lengths]

    def __iter__(self):
        for i, sampler in enumerate(self.samplers):
            for idx in sampler:
                yield self.start_idx[i] + idx


class GlobalGroupRandomSampler(Sampler):
    """
    Ensure all the data within a global batch are from the same group.
    For those groups which can not be divided by global_batchsize, we over sample them
    """
    def __init__(self, global_batchsize: int, modality_group_indices: List[List[int]]):
        """
        Initialize the sampler.

        Args:
        - global_batchsize (int): The size of the global batch.
        - modality_group_indices (List[List[int]]): A list of lists containing indices for each group.
        """
        self.global_batchsize = global_batchsize
        self.modality_group_indices = modality_group_indices
        self._calculate_lengths()
        self._calculate_start_indices()
        self.indices = []
        self._prepare_indices()
        random.shuffle(self.indices)

    def _calculate_lengths(self):
        self.lengths = [len(group) for group in self.modality_group_indices]

    def _calculate_start_indices(self):
        self.start_idx = [0]
        for length in self.lengths:
            self.start_idx.append(self.start_idx[-1] + length)

    def _prepare_indices(self):
        """
        Prepare the list of indices to sample from, ensuring that each batch contains indices from the same group.
        Oversample groups that do not fit the global batch size perfectly.
        """
        if not self.lengths:
            raise ValueError("Lengths must be provided.")

        for group_idx, length in enumerate(self.lengths):
            group_indices = self.modality_group_indices[group_idx]
            random.shuffle(group_indices)

            num_batches = (length + self.global_batchsize - 1) // self.global_batchsize
            for _ in range(num_batches):
                batch_indices = self._get_batch_indices(group_indices)
                self.indices.append(batch_indices)

    def _get_batch_indices(self, group_indices):
        batch_indices = group_indices[:self.global_batchsize]
        if len(batch_indices) < self.global_batchsize:
            # If the batch is smaller than the global batch size, oversample
            batch_indices += random.choices(group_indices, k=self.global_batchsize - len(batch_indices))
        return batch_indices

    def __iter__(self):
        flat_indices = [idx for batch in self.indices for idx in batch]
        return iter(flat_indices)

    def __len__(self):
        return sum(len(batch) for batch in self.indices)


if __name__ == "__main__":
    sampler = GlobalGroupRandomSampler(6, [[12, 15, 13, 16], [10, 20, 30, 40, 50], [99, 88, 77, 66, 55, 44, 33, 22, 11]])
    print(len(sampler))
    for i, idx in enumerate(sampler):
        print(i, ": ", idx)
