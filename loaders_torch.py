"""Contains loaders for torch structures."""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import (
    pad_sequence,  
    pack_padded_sequence,
    PackedSequence
)
from constants import BATCH_SIZE


class CustomDataset(Dataset):
    """Dataset class required by torch dataloader."""
    def __init__(self, X: np.ndarray, y: np.ndarray, device: str = "CPU"):
        self._X = X 
        self._y = y

        if device not in ["GPU", "CPU"]:
            raise ValueError("Device must be 'GPU' or 'CPU'.")
        if device == "GPU":
            self._device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        else:
            self._device = torch.device("cpu")

        self._X_tensor = [torch.tensor(x, dtype=torch.float32, device=self._device) for x in self._X]
        self._y_tensor = torch.tensor(self._y).to(self._device)

    def __len__(self):
        return len(self._X)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:     
        return self._X_tensor[idx], self._y_tensor[idx]

def custom_pad_collate(
        batch: list, 
    ) -> tuple[PackedSequence, torch.Tensor]:
    """
    Func does non-naive batching of time series data.
    Ultimately, it returns a PackedSequence neeeded
    as time-series of variable lengths.
    """
    batch.sort(key=lambda x: len(x[0]), reverse=True)   # Descending order; required by pack_padded_sequence 

    # Cross-entropy loss API requires target and logits as tensors
    labels = torch.stack([item[1] for item in batch])
    mfccs = [item[0] for item in batch]
    lengths = [len(item[0]) for item in batch]

    padded_mfccs = pad_sequence(mfccs, batch_first=True) 
    packed_mfccs = pack_padded_sequence(
        padded_mfccs, 
        lengths, 
        batch_first=True
    )
    return packed_mfccs, labels

def create_dataloader(
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int = BATCH_SIZE, 
        shuffle: bool = True,
        device: str = "CPU",
        ) -> DataLoader:
    """Returns a data loader."""
    if batch_size > 1:
        collate_fn = custom_pad_collate
    else:
        collate_fn = None

    data = CustomDataset(X, y, device)
    dataloader = DataLoader(
        data, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        collate_fn=collate_fn,
    )

    return dataloader 
