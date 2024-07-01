import torch
import numpy as np
from typing import Dict


class IMDBDataset:
    def __init__(self, reviews: np.array, targets: np.array):
        """
        :param reviews: this is a numpy array
        :param targets: a vector, numpy array
        """
        self.reviews = reviews
        self.targets = targets

    def __len__(self) -> int:
        # returns length of the dataset
        return len(self.reviews)

    def __getitem__(self, item) -> Dict[str, torch.tensor]:
        # for any given item, which is an int,
        # return review and targets as torch tensor
        # item is the index of the item in concern
        review = self.reviews[item, :]
        target = self.targets[item]

        return {
            "review": torch.tensor(review, dtype=torch.long),
            "target": torch.tensor(target, dtype=torch.float),
        }
