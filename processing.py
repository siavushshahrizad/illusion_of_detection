"""File contains various funcs to process data."""

import torch
import numpy as np
import pandas as pd
from constants import MFCC


def compute_mean_and_std(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Compute global mean and for MFCCs. Assumes data is in shape (t, mfcc).""" 
    stacked = np.vstack(df[MFCC], dtype=np.float32)
    means = np.mean(stacked, axis=0)
    stds = np.std(stacked, ddof=1, axis=0)
    return means, stds

def normalise_df(df: pd.DataFrame, mean: np.ndarray, std: np.ndarray) -> None:
    """Normalises MFCCS. Assumes data is in shape (t, mfcc)."""
    df[MFCC] = df[MFCC].apply(lambda x: (x - mean) / std)

def compress_left(seq: np.ndarray) -> np.ndarray:
    """
    Averages left side of a time series of n MFCCs.
    Assumes data is in shape (t, mfcc).
    """
    length = seq.shape[0]
    mid = length // 2
    half = seq[:mid, :]
    compressed = half.mean(axis=0)
    return compressed

def compress_right(seq: np.ndarray) -> np.ndarray:
    """
    Averages right side of a time series of n MFCCs.
    Assumes data is in shape (t, mfcc).
    """
    length = seq.shape[0]
    mid = length // 2
    half = seq[mid:, :]
    compressed = half.mean(axis=0)
    return compressed

def compute_class_weights(df: pd.DataFrame, outcome: str) -> torch.Tensor:
    """
    Returns weights for CE loss, or model doesn't learn 
    with imbalanced data.
    """
    counts = df[outcome].value_counts().sort_index()            # Majority class first in counts
    num_classes = len(counts)
    total = len(df)
    weights = total / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)           # Model requires tensor.float32 or crash
