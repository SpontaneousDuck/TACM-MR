"""
TACM Dataset - Topographically-Augmented Channel Model Dataset

A PyTorch Lightning compatible dataset for automatic modulation recognition research
that combines realistic terrain-based channel models with modulated signals.

Usage:
    >>> from tacm_dataset import TACMDataModule
    >>> 
    >>> dm = TACMDataModule(
    ...     dataset_root="~/data/tacm",
    ...     batch_size=32,
    ...     download=True,
    ...     gen_chan=True
    ... )
    >>> dm.prepare_data()
    >>> dm.setup()
    >>> train_loader = dm.train_dataloader()
"""

from .TACMDataModule import (
    TACMDataModule,
    TACMDataset, 
    TACMSampler,
    TACMSubset
)

from .scenario_gen import ScenarioGenerator

__version__ = "1.0.0"

__all__ = [
    "TACMDataModule",
    "TACMDataset",
    "TACMSampler", 
    "TACMSubset",
    "ScenarioGenerator"
]