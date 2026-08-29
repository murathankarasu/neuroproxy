"""Dataset adapters, all exposing the same `Recording` contract."""
from __future__ import annotations

from .base import Dataset, Recording, available_datasets, get_dataset, resample_to
from .mcd_rppg import MCDrPPG
from .pure import PURE
from .scamps import SCAMPS
from .synthetic import (
    SyntheticConfig,
    SyntheticDataset,
    SyntheticProtocolDataset,
    SyntheticStressDataset,
    generate,
    window_condition,
)
from .ubfc_phys import UBFCPhys
from .ubfc_rppg import UBFCrPPG

__all__ = [
    "Dataset",
    "Recording",
    "available_datasets",
    "get_dataset",
    "resample_to",
    "SyntheticDataset",
    "SyntheticStressDataset",
    "SyntheticProtocolDataset",
    "window_condition",
    "SyntheticConfig",
    "generate",
    "UBFCrPPG",
    "UBFCPhys",
    "MCDrPPG",
    "PURE",
    "SCAMPS",
]
