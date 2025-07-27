"""
TACM Dataset - Topographically-Augmented Channel Model Dataset for Automatic Modulation Recognition

This module provides PyTorch Lightning data modules and utilities for working with the TACM dataset,
which combines realistic terrain-based channel models with modulated signals for machine learning applications.

The TACM dataset extends the CSPB.ML.2018R2 dataset by applying realistic channel effects derived from 
terrain-based propagation models. This enables more realistic evaluation of automatic modulation 
recognition systems under diverse propagation conditions.

Classes:
    TACMDataset: Main dataset class for loading and accessing TACM data
    TACMSampler: Custom sampler for mixed signal-channel batch sampling
    TACMSubset: Subset class for train/validation/test splits
    TACMDataModule: PyTorch Lightning DataModule for TACM dataset

Example:
    >>> from tacm_dataset import TACMDataModule
    >>> dm = TACMDataModule(
    ...     dataset_root="/path/to/data",
    ...     batch_size=32,
    ...     download=True,  # Download base dataset on first use
    ...     gen_chan=True   # Generate channel files on first use
    ... )
    >>> dm.prepare_data()
    >>> dm.setup()
    >>> train_loader = dm.train_dataloader()
"""

from typing import Iterator, List, Sequence
import pytorch_lightning as pl
import h5py
import os
import hashlib
import urllib.request
import zipfile
import glob
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import Sampler, Dataset, DataLoader
from tqdm import tqdm

from .sionna_torch.ApplyTimeChannel import ApplyTimeChannel
from .scenario_gen import ScenarioGenerator

class TACMDataset:
    """
    Dataset class for the TACM (Topographically-Augmented Channel Model) dataset.
    
    This dataset combines modulated signals from the CSPB.ML.2018R2 dataset with
    realistic channel effects generated using terrain-based models.
    
    Args:
        dataset_root: Root directory containing the dataset files
        frame_size: Size of each signal frame (default: 1024, must divide 32768 evenly)
        chan_file: Optional path to specific channel file
        scenario: Dataset scenario type ('a', 'b', 'c', or 'd')
        n_rx: Number of receivers (default: 1)
        map_res: Map resolution in meters/pixel (default: 20)
        
    The dataset supports different scenarios:
        - Scenario 'a': Standard single-receiver setup
        - Scenario 'b': Multi-receiver with subsampling
        - Scenario 'c': Multi-receiver flattened format
        - Scenario 'd': Multi-receiver with variable receiver count
        
    Raises:
        RuntimeError: If frame_size does not evenly divide the original frame size (32768)
        FileNotFoundError: If required dataset files are not found
    """

    def __init__(self, 
                 dataset_root: str, 
                 frame_size: int = 1024, 
                 chan_file: str = None,
                 scenario: str = 'a',
                 n_rx: int = 1, 
                 map_res: float = 20,
    ) -> None:

        print('Loading CSPB.ML.2018R2 signal data...')
        cspb_path = os.path.join(dataset_root, "CSPB.ML.2018R2_nonoise.hdf5")
        if not os.path.exists(cspb_path):
            raise FileNotFoundError(f"CSPB dataset file not found: {cspb_path}")
            
        with h5py.File(cspb_path, "r") as f:
            x = torch.from_numpy(f['x'][()])
            y = torch.from_numpy(f['y'][()]).to(torch.long)
            t0 = torch.from_numpy(f['T0'][()]).to(torch.long)
            beta = torch.from_numpy(f['beta'][()])
            T_s = torch.from_numpy(f['T_s'][()])
            
        print("Preprocessing signal data...")
        S_idx = torch.linspace(0, 1.0, x.shape[-1]).repeat(x.shape[0], 1)
        S_idx *= (32768/T_s[:,None])
        S_idx = S_idx.floor_().short()

        # Validate and reshape data to frame_size
        if 32768 % frame_size:
            raise RuntimeError(f"frame_size ({frame_size}) is not an even divisor of the original frame size (32768).")
            
        x = x.reshape((-1, 1, frame_size))
        reshape_factor = 32768//frame_size
        y = torch.repeat_interleave(y, reshape_factor)
        t0 = torch.repeat_interleave(t0, reshape_factor)
        beta = torch.repeat_interleave(beta, reshape_factor)
        T_s = torch.repeat_interleave(T_s, reshape_factor)
        S_idx = S_idx.reshape((-1, frame_size))
        S_idx -= S_idx[:,:1]

        # Scale signal to 0dB power
        transmit_power = 0  # 0dB
        signal_power = torch.mean(torch.abs(x) ** 2, -1)[...,None]
        target_snr_linear = 10 ** (transmit_power / 10)
        signal_scale_linear = torch.sqrt(target_snr_linear / signal_power)
        x = x * signal_scale_linear
        
        # Load channel data
        print('Loading channel data...')
        hdf5_path = chan_file if chan_file is not None else os.path.join(dataset_root, f"TACM_2025_1_channels_{scenario}_{n_rx if scenario == 'd' else 8}rx.h5py")
        if not os.path.exists(hdf5_path):
            raise FileNotFoundError(f"Channel file {hdf5_path} does not exist. Set gen_chan=True to generate channel files.")
            
        with h5py.File(hdf5_path, "r") as f:
            h_t = torch.from_numpy(f['h_t'][()])
            tx_xy = torch.from_numpy(f['tx_xy'][()])
            rx_xy = torch.from_numpy(f['rx_xy'][()])
            topography = torch.from_numpy(f['topography'][()]) 

        # Process channel data based on scenario
        h_t = h_t[:,:n_rx]
        tx_xy = tx_xy[:,:n_rx]
        tx_xy[..., :2] *= map_res
        rx_xy = rx_xy[:,:n_rx]
        rx_xy[..., :2] *= map_res

        # Store processed data
        self.data_x = x
        self.data_y = y
        self.data_T_s = T_s
        self.data_beta = beta
        self.data_S_idx = S_idx
        
        # Configure data layout based on scenario
        if scenario == "a" or scenario == "d":
            # Standard setup - signals and channels paired directly
            self.data_h_t = h_t
            self.data_tx_xy = tx_xy
            self.data_rx_xy = rx_xy
            self.topography = topography
        elif scenario == "c":
            # Flattened multi-receiver format
            self.data_h_t = h_t[:, :n_rx].flatten(0,1).unsqueeze(1)
            self.data_tx_xy = tx_xy[:].repeat(n_rx, 1, 1)
            self.data_rx_xy = rx_xy[:, :n_rx].flatten(0,1).unsqueeze(1)
            self.topography = topography.repeat(n_rx, 1, 1)
        elif scenario == "b":
            # Multi-receiver with signal subsampling
            self.data_x = x[::n_rx]
            self.data_y = y[::n_rx]
            self.data_T_s = T_s[::n_rx]
            self.data_beta = beta[::n_rx]
            self.data_S_idx = S_idx[::n_rx]
            self.data_h_t = h_t[::n_rx, :n_rx]
            self.data_tx_xy = tx_xy[::n_rx]
            self.data_rx_xy = rx_xy[::n_rx, :n_rx]
            self.topography = topography[::n_rx]
        else:
            raise ValueError(f"Unknown scenario '{scenario}'. Must be one of: 'a', 'b', 'c', 'd'")

    def __getitem__(self, index):
        """
        Retrieve a single data sample.
        
        Args:
            index: Tuple of (signal_index, channel_index)
            
        Returns:
            Dictionary containing:
                - 'x': Signal data
                - 'h_t': Channel response
                - 'y': Modulation class label
                - 'T_s': Symbol period
                - 'beta': Roll-off factor
                - 'S_idx': Symbol indices
                - 'tx_xy': Transmitter coordinates
                - 'rx_xy': Receiver coordinates
        """
        x = self.data_x[index[0]]
        h_t = self.data_h_t[index[1]]
        return {
            'x': x, 
            'h_t': h_t, 
            'y': self.data_y[index[0]], 
            'T_s': self.data_T_s[index[0]], 
            'beta': self.data_beta[index[0]], 
            'S_idx': self.data_S_idx[index[0]], 
            'tx_xy': self.data_tx_xy[index[1]], 
            'rx_xy': self.data_rx_xy[index[1]]
        }

    def __len__(self):
        """Return the number of signal samples in the dataset."""
        return len(self.data_x)

class TACMSampler(Sampler[List]):
    """
    Custom sampler that returns mixed batches of signal and channel samples.
    
    This sampler ensures that each batch contains pairs of signal indices and
    channel indices, allowing for random combinations of signals and channels
    during training. This is essential for the TACM dataset where signals and
    channels are independently sampled.
    
    Args:
        x_len: Number of signal samples
        h_len: Number of channel samples  
        batch_size: Size of mini-batch
        generator: Random number generator for reproducibility
        
    Raises:
        ValueError: If batch_size is not a positive integer
    """

    def __init__(self, x_len: int, h_len: int, batch_size: int, generator=None) -> None:
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError(f"batch_size should be a positive integer value, but got batch_size={batch_size}")
        self.x_len = x_len
        self.h_len = h_len
        self.batch_size = batch_size
        if generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            self.generator = torch.Generator().manual_seed(seed)
        else:
            self.generator = generator

    def _sampler_iter(self) -> Iterator[int]:
        """Generate random pairs of (signal_index, channel_index)."""
        x_iter = torch.randperm(self.x_len, generator=self.generator).tolist()
        h_perm = torch.randperm(self.h_len, generator=self.generator)
        repeat_factor = self.x_len // self.h_len
        h_perm = h_perm.repeat(repeat_factor).tolist()

        yield from zip(x_iter, h_perm)

    def __iter__(self) -> Iterator[List]:
        """Iterate over batches of (signal_index, channel_index) pairs."""
        batch = [(0,0)] * self.batch_size
        idx_in_batch = 0
        for idx in self._sampler_iter():
            batch[idx_in_batch] = idx
            idx_in_batch += 1
            if idx_in_batch == self.batch_size:
                yield batch
                idx_in_batch = 0
                batch = [(0,0)] * self.batch_size
        if idx_in_batch > 0:
            yield batch[:idx_in_batch]

    def __len__(self) -> int:
        """Return the number of batches."""
        return (self.x_len + self.batch_size - 1) // self.batch_size

class TACMSubset(Dataset):
    """
    Subset of a TACM dataset at specified indices.
    
    This class allows for creating train/validation/test splits of the dataset
    while maintaining separate indexing for signals and channels.
    
    Args:
        dataset: The parent TACMDataset
        x_indices: Indices for signal samples
        h_indices: Indices for channel samples
    """

    dataset: TACMDataset
    x_indices: Sequence[int]
    h_indices: Sequence[int]

    def __init__(self, dataset: TACMDataset, x_indices: Sequence[int], h_indices: Sequence[int]) -> None:
        self.dataset = dataset
        self.x_indices = x_indices
        self.h_indices = h_indices

    def __getitem__(self, idx):
        """
        Retrieve a sample from the subset.
        
        Args:
            idx: Index or list of (signal_index, channel_index) pairs
            
        Returns:
            Sample data from the parent dataset
        """
        if isinstance(idx, list):
            return self.dataset[[(self.x_indices[x_i], self.h_indices[h_i]) for x_i, h_i in idx]]
        return self.dataset[(self.x_indices[idx[0]], self.h_indices[idx[1]])]

    def __len__(self):
        """Return the number of samples in this subset."""
        return len(self.x_indices)

# Download URLs and MD5 checksums for the CSPB.ML.2018R2 noise-free dataset files
CSPB2018R2_NONOISE_META = [
    ('https://cyclostationary.blog/wp-content/uploads/2025/04/CSPB.ML_.2018R2_1_noisefree.zip', '70a9f6c207132f4a4dc5c14ef188b30a'), 
    ('https://cyclostationary.blog/wp-content/uploads/2025/04/CSPB.ML_.2018R2_2_noisefree.zip', '33c8fbf8fa533aae6bf0cd8e07597dbd'), 
    ('https://cyclostationary.blog/wp-content/uploads/2025/04/CSPB.ML_.2018R2_3_noisefree.zip', '38a53553e35499e4a64a59721c6a6f6b'), 
    ('https://cyclostationary.blog/wp-content/uploads/2025/04/CSPB.ML_.2018R2_4_noisefree.zip', 'a370063bf9091c5d44874880eb6861b2'), 
    ('https://cyclostationary.blog/wp-content/uploads/2025/04/CSPB.ML_.2018R2_5_noisefree.zip', 'f412ca01c7a1021f575089759d389edb'), 
    ('https://cyclostationary.blog/wp-content/uploads/2025/04/CSPB.ML_.2018R2_6_noisefree.zip', 'f726da3ed68b3fa1439e36d71aa21ced'), 
    ('https://cyclostationary.blog/wp-content/uploads/2025/04/CSPB.ML_.2018R2_7_noisefree.zip', '071b7a9825c3c83594960e1ccb94f41a'), 
    ('https://cyclostationary.blog/wp-content/uploads/2025/04/CSPB.ML_.2018R2_8_noisefree.zip', 'ac3d2575ecc243343610f3064e60c1c7'), 
    ('https://cyclostationary.blog/wp-content/uploads/2025/04/CSPB.ML_.2018R2_9_noisefree.zip', '14075f75a01804a30d386e0a2dcf9bf1'), 
    ('https://cyclostationary.blog/wp-content/uploads/2025/04/CSPB.ML_.2018R2_10_noisefree.zip', '34a9a343c7abc249c6e6e6d7a099e9c2'), 
    ('https://cyclostationary.blog/wp-content/uploads/2025/04/CSPB.ML_.2018R2_11_noisefree.zip', 'fc0d7b719f22789d2cb19bdb0df4c6b9'), 
    ('https://cyclostationary.blog/wp-content/uploads/2025/04/CSPB.ML_.2018R2_12_noisefree.zip', 'a8f9bfea9a73c28968340e0642cdc017'), 
    ('https://cyclostationary.blog/wp-content/uploads/2025/04/CSPB.ML_.2018R2_13_noisefree.zip', 'f7aea4a4cb5d7eceb323db20c495ec1e'), 
    ('https://cyclostationary.blog/wp-content/uploads/2025/04/CSPB.ML_.2018R2_14_noisefree.zip', 'cbb519058aa7de2f92c34d2e44a575e6'), 
    ('https://cyclostationary.blog/wp-content/uploads/2025/04/CSPB.ML_.2018R2_15_noisefree.zip', '1eefb310dc4c0e1db2a16981de0973a2'), 
    ('https://cyclostationary.blog/wp-content/uploads/2025/04/CSPB.ML_.2018R2_16_noisefree.zip', '06efabc0a87e655787eede6a6d08a982'), 
    ('https://cyclostationary.blog/wp-content/uploads/2025/04/CSPB.ML_.2018R2_17_noisefree.zip', '6dc91bf8857fc2da31c449ef1f29af2a'), 
    ('https://cyclostationary.blog/wp-content/uploads/2025/04/CSPB.ML_.2018R2_18_noisefree.zip', '7776938bb45c8292ef7b00a30bf7b896'), 
    ('https://cyclostationary.blog/wp-content/uploads/2025/04/CSPB.ML_.2018R2_19_noisefree.zip', '31e602f7af2b2918bf9100a61eac2f3c'), 
    ('https://cyclostationary.blog/wp-content/uploads/2025/04/CSPB.ML_.2018R2_20_noisefree.zip', '3f9d7c2c830d7a7c2099e60174ae1cee'), 
    ('https://cyclostationary.blog/wp-content/uploads/2025/04/CSPB.ML_.2018R2_21_noisefree.zip', '46283d587cc17ec0fcd4a33fecaea9a2'), 
    ('https://cyclostationary.blog/wp-content/uploads/2025/04/CSPB.ML_.2018R2_22_noisefree.zip', 'c091bfca1ac47d12127ee6da20be6dde'), 
    ('https://cyclostationary.blog/wp-content/uploads/2025/04/CSPB.ML_.2018R2_23_noisefree.zip', 'e549163eb54b6f3c97e55530df082d43'), 
    ('https://cyclostationary.blog/wp-content/uploads/2025/04/CSPB.ML_.2018R2_24_noisefree.zip', '37f6d463b6f066898efec82c614f5b61'), 
    ('https://cyclostationary.blog/wp-content/uploads/2025/04/CSPB.ML_.2018R2_25_noisefree.zip', '04545743999b238c9fea1a5685b8be84'), 
    ('https://cyclostationary.blog/wp-content/uploads/2025/04/CSPB.ML_.2018R2_26_noisefree.zip', 'e54358efdcc597b1228d04824663fd3c'), 
    ('https://cyclostationary.blog/wp-content/uploads/2025/04/CSPB.ML_.2018R2_27_noisefree.zip', '3beaca45b90488e53d8c84f63dd86883'), 
    ('https://cyclostationary.blog/wp-content/uploads/2025/04/CSPB.ML_.2018R2_28_noisefree.zip', '12945c4d0929ed40df2d2f16fb02f30f'), 
    ('https://cyclostationary.blog/wp-content/uploads/2023/09/signal_record_C_2023.txt', '1f29b1a9fd4b480118e9734bc06126e1'), 
]

class TACMDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for the TACM dataset.
    
    This DataModule handles data preparation, downloading, channel generation,
    and provides train/validation/test dataloaders for the TACM dataset.
    
    Args:
        dataset_root: Root directory where dataset files will be stored
        batch_size: Batch size for dataloaders
        download: Whether to download the base CSPB dataset if missing (default: False)
        gen_chan: Whether to generate new channel files (default: False)
        n_topo: Number of topographic scenarios to generate (default: 28000)
        in_memory: Whether to keep data in memory (default: True)
        frame_size: Size of signal frames in samples (default: 1024)
        f_c: Carrier frequency in Hz (default: 900e6)
        bw: Signal bandwidth in Hz (default: 30e3)
        reuse_factor: Terrain reuse factor for efficiency (default: 128)
        gen_batch_size: Batch size for channel generation (default: 128)
        snr_inband: Whether to use in-band SNR calculation (default: True)
        snr_range: SNR range in dB as [min, max] (default: [-30, 10])
        scenario: Dataset scenario type ('a', 'b', 'c', or 'd') (default: 'a')
        seed: Random seed for reproducibility (default: 43)
        n_rx: Number of receivers (default: 8)
        n_workers: Number of data loading workers (default: 8)
        
    Scenarios:
        - 'a': Standard single-receiver setup
        - 'b': Multi-receiver with signal subsampling  
        - 'c': Multi-receiver flattened format
        - 'd': Multi-receiver with variable receiver count
        
    Example:
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
    
    def __init__(self, 
                 dataset_root: str, 
                 batch_size: int, 
                 download: bool = False,
                 gen_chan: bool = False,
                 n_topo: int = 28000,
                 in_memory: bool = True,
                 frame_size: int = 1024, 
                 f_c: float = 900e6, 
                 bw: float = 30e3, 
                 reuse_factor: int = 128,
                 gen_batch_size: int = 128,
                 snr_inband: bool = True, 
                 snr_range: List[int] = [-30, 10],
                 scenario: str = 'a', 
                 seed: int = 43, 
                 n_rx: int = 8, 
                 n_workers: int = 8):
        super().__init__()
        
        # Validate inputs
        if scenario not in ['a', 'b', 'c', 'd']:
            raise ValueError(f"Invalid scenario '{scenario}'. Must be one of: 'a', 'b', 'c', 'd'")
        if frame_size <= 0 or 32768 % frame_size != 0:
            raise ValueError(f"frame_size ({frame_size}) must be positive and divide 32768 evenly")
        if batch_size <= 0:
            raise ValueError(f"batch_size ({batch_size}) must be positive")
        if n_rx <= 0:
            raise ValueError(f"n_rx ({n_rx}) must be positive")
            
        self.dataset_root = os.path.expanduser(dataset_root)
        self.batch_size = batch_size if scenario != "c" else batch_size//n_rx
        self.download = download
        self.gen_chan = gen_chan
        self.in_memory = in_memory
        self.terrain_reuse_factor = reuse_factor
        self.gen_batch_size = gen_batch_size
        self.frame_size = frame_size
        self.transforms = []
        self.seed = seed
        self.n_rx = n_rx
        self.n_topo = n_topo
        self.f_c = f_c
        self.bw = bw
        self.map_res = 20  # meters/pixel
        self.map_size = 1000  # Pixels x Pixels map size
        self.n_workers = n_workers
        self.snr_inband = snr_inband
        self.snr_range = snr_range
        self.scenario = scenario

        # Calculate noise power
        noise_power_db = -173.8 + 10 * np.log10(bw)
        self.noise_power_lin = 10**((noise_power_db)/10)
       
        # Define modulation classes
        self.classes = ['bpsk', 'qpsk', '8psk', 'dqpsk', 'msk', '16qam', '64qam', '256qam']
        self.ds_train, self.ds_val, self.ds_test = [], [], []


    def prepare_data(self):
        """
        Download and prepare the base dataset files.
        
        This method:
        1. Downloads the CSPB.ML.2018R2 noise-free dataset if not present
        2. Extracts and processes the signal files
        3. Creates a consolidated HDF5 file with signals and metadata
        4. Generates channel files using terrain-based models if requested
        
        Note: This method is called only once per node in distributed training.
        """ 
        # Check if CSPB dataset needs to be downloaded and processed
        cspb_path = os.path.join(self.dataset_root, "CSPB.ML.2018R2_nonoise.hdf5")
        if self.download or not os.path.exists(cspb_path):
            print("Downloading and processing CSPB.ML.2018R2 dataset...")
            self._download_and_process_cspb()

        # Check if channel files need to be generated
        channel_path = os.path.join(self.dataset_root, f"TACM_2025_1_channels_{self.scenario}_{self.n_rx if self.scenario == 'd' else 8}rx.h5py")
        if self.gen_chan or not os.path.exists(channel_path):
            print("Generating channel files...")
            self._generate_channel_files(channel_path)

    def _download_and_process_cspb(self):
        """Download and process the CSPB.ML.2018R2 dataset."""
        download_root = os.path.join(self.dataset_root, "CSPB_ML_2018R2_nonoise_zips")
        extract_root = os.path.join(self.dataset_root, "CSPB_ML_2018R2_nonoise_signals")
        os.makedirs(download_root, exist_ok=True)
        os.makedirs(extract_root, exist_ok=True)

        # Download and extract files
        for url, zip_md5 in CSPB2018R2_NONOISE_META:
            fname = os.path.basename(url)
            if 'zip' in fname:
                fpath = os.path.join(download_root, fname)
            else: 
                fpath = os.path.join(extract_root, fname)

            # Download file if it doesn't exist
            if not os.path.exists(fpath):
                print(f"Downloading {url} to {fpath}")
                with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "pytorch/vision"})) as response:
                    content = iter(lambda: response.read(1024*32), b"")
                    with open(fpath, "wb") as fh:
                        for chunk in content:
                            if not chunk:
                                continue
                            fh.write(chunk)

            # Verify file integrity
            md5_alg = hashlib.md5(usedforsecurity=False)
            with open(fpath, "rb") as f:
                for chunk in iter(lambda: f.read(1024*1024), b""):
                    md5_alg.update(chunk)
            if zip_md5 != md5_alg.hexdigest():
                raise RuntimeError(f"File not found or corrupted: {fpath}")

            # Extract zip files
            if 'zip' in fname:
                print(f"Extracting {fpath} to {extract_root}")
                with zipfile.ZipFile(fpath, "r", compression=zipfile.ZIP_STORED) as zipf:
                    zipf.extractall(extract_root)

        # Process signal files
        print("Processing signal files...")
        fnames = glob.glob(os.path.join(extract_root, "*", "signal*.tim"))
        fnames = sorted(fnames, key=lambda x: int(Path(x).stem.split("_")[-1]))

        x = np.array([np.fromfile(fname)[1:].view(np.complex64) for fname in fnames])

        # Load and process labels
        labels = pd.read_csv(os.path.join(extract_root, "signal_record_C_2023.txt"), 
                           header=None, sep='\s+', index_col=0, 
                           names=['label', 'T0', 'cfo', 'beta', 'U', 'D', 'SNR', 'P(N)'])
        y = np.array([self.classes.index(label) for label in labels.label])

        # Set all MSK to a beta of 0.5 for consistent bandwidth
        labels.loc[labels['label'] == "msk", 'beta'] = 0.5

        # Extract parameters
        t0 = labels.T0.to_numpy()
        d = labels.D.to_numpy()
        u = labels.U.to_numpy()

        # Save processed data to HDF5
        print(f"Saving processed data to {os.path.join(self.dataset_root, 'CSPB.ML.2018R2_nonoise.hdf5')}")
        with h5py.File(os.path.join(self.dataset_root, "CSPB.ML.2018R2_nonoise.hdf5"), "w") as f:
            f['x'] = x
            f['y'] = y
            f['T0'] = t0
            f['cfo'] = labels.cfo.to_numpy()
            f['beta'] = labels.beta.to_numpy()
            f['U'] = u
            f['D'] = d
            f['T_s'] = 1 / ((1/t0)*(d/u))

    def _generate_channel_files(self, channel_path: str):
        """Generate channel files using terrain-based models."""
        batch_size = 128
        n_rx = self.n_rx if self.scenario == "d" else 8
        dev = self.trainer.strategy.root_device if self.trainer is not None else torch.get_default_device()
        
        scenario_gen = ScenarioGenerator(
            n_receivers=n_rx, 
            batch_size=batch_size, 
            map_size=self.map_size, 
            map_resolution=self.map_res, 
            min_receiver_dist=2, 
            max_iter=400, 
            frame_size=1,
            f_c=self.f_c,
            bw=self.bw,
            noise_type='perlin',
            seed=self.seed,
            target_total_p=False, 
            dtype=torch.float32, 
            device=dev
        )

        # Pre-allocate arrays for generated data
        h_t = torch.empty((self.n_topo, n_rx, 1, 1, 1, 1, 14), device=torch.device('cpu'), dtype=torch.complex64)
        topography = torch.empty(self.n_topo, self.map_size, self.map_size, device=torch.device('cpu'))
        tx_xy = torch.empty(self.n_topo, 1, 3, device=torch.device('cpu'))
        rx_xy = torch.empty(self.n_topo, n_rx, 3, device=torch.device('cpu'))
        
        snr_gen = torch.Generator().manual_seed(self.seed)
        snr_min, snr_max = self.snr_range[0], self.snr_range[1]
        target_snr = torch.empty(n_rx)

        # Generate channel scenarios
        for i in tqdm(range(0, self.n_topo), desc="Generating channels", miniters=100, mininterval=10):
            target_snr = target_snr.uniform_(snr_min, snr_max, generator=snr_gen)
            
            if self.scenario == "d":
                target_power = target_snr + 10*np.log10((10**(scenario_gen.chan_gen.get_noise_power()/10))*n_rx)
                target_power[:] = 10*torch.log10((10**(target_power[0]/10))/n_rx)
            else:
                target_power = target_snr + scenario_gen.chan_gen.get_noise_power()
                 
            h_t[i] = scenario_gen.RegenerateFullScenario(target_power)
            tx_xy[i] = scenario_gen.transmitters
            rx_xy[i] = scenario_gen.receivers
            topography[i] = scenario_gen.map

        print(f"Writing channel data to: {channel_path}")
        with h5py.File(channel_path, "w") as f:
            f.create_dataset('h_t', data=h_t.numpy(force=True), compression="gzip")
            f.create_dataset('tx_xy', data=tx_xy, compression="gzip")
            f.create_dataset('rx_xy', data=rx_xy, compression="gzip")
            f.create_dataset('topography', data=topography, compression="gzip")

    def setup(self, stage: str = None):
        """
        Set up the dataset splits and channel application.
        
        This method:
        1. Creates the main dataset object
        2. Splits data into train/validation/test sets (60%/20%/20%)
        3. Initializes channel application for realistic propagation effects
        
        Args:
            stage: Optional stage identifier ('fit', 'test', etc.)
        """
        if not len(self.ds_train) or not len(self.ds_val) or not len(self.ds_test):
            self.generator = torch.Generator().manual_seed(self.seed)
            
            # Set up channel processing parameters
            self.l_min, self.l_max = -6, int(np.ceil(3e-6*self.bw)) + 6
            l_tot = self.l_max-self.l_min+1
            dev = self.trainer.strategy.root_device if self.trainer is not None else torch.get_default_device()
            self.snr_gen = torch.Generator(dev).manual_seed(self.seed)
            self._apply_channel = ApplyTimeChannel(self.frame_size, l_tot=l_tot, rng=self.snr_gen, add_awgn=True, device=dev)

            # Create main dataset
            self.ds = TACMDataset(
                self.dataset_root,
                frame_size=self.frame_size,
                n_rx=self.n_rx,
                scenario=self.scenario,
                map_res=self.map_res,
            )

            # Create train/validation/test splits (60%/20%/20%)
            subset_lengths = [0.6, 0.2, 0.2]
            
            # Split signal indices
            subset_lengths_x = [int(len(self.ds.data_x) * frac) for frac in subset_lengths]
            remainder = len(self.ds) - sum(subset_lengths_x)
            # Distribute remainder across splits in round-robin fashion
            for i in range(remainder):
                idx_to_add_at = i % len(subset_lengths_x)
                subset_lengths_x[idx_to_add_at] += 1
            indices_x = torch.randperm(sum(subset_lengths_x), generator=self.generator).tolist()

            # Split channel indices
            subset_lengths_h = [int(len(self.ds.data_h_t) * frac) for frac in subset_lengths]
            remainder = len(self.ds.data_h_t) - sum(subset_lengths_h)
            # Distribute remainder across splits in round-robin fashion
            for i in range(remainder):
                idx_to_add_at = i % len(subset_lengths_h)
                subset_lengths_h[idx_to_add_at] += 1
            indices_h = torch.randperm(sum(subset_lengths_h), generator=self.generator).tolist()

            # Create dataset subsets
            self.ds_train, self.ds_val, self.ds_test = [
                TACMSubset(self.ds, indices_x[offset_x - length_x : offset_x], indices_h[offset_h - length_h : offset_h])
                for offset_x, length_x, offset_h, length_h 
                in zip(np.cumsum(subset_lengths_x).tolist(), subset_lengths_x, np.cumsum(subset_lengths_h).tolist(), subset_lengths_h)
            ]

    def on_after_batch_transfer(self, batch, dataloader_idx):
        """
        Apply channel effects and prepare batch data.
        
        This method:
        1. Applies realistic channel effects to signals
        2. Adds noise based on SNR settings
        3. Normalizes signals to [-1, 1] range
        4. Calculates SNR metrics (per-receiver and total)
        
        Args:
            batch: Batch of data samples
            dataloader_idx: Index of the dataloader
            
        Returns:
            Processed batch with channel effects applied
        """
        if isinstance(batch, dict):
            x = batch['x']
            h_t = batch['h_t']

            # Apply channel and noise
            z, rx_pow_db, snr = self._apply_channel(x[:,:1,None], h_t, self.noise_power_lin, None)
            z = z.squeeze(2,3)[...,self.l_min*-1:self.l_max*-1]

            # Normalize to [-1.0, 1.0] range per frame
            new_min, new_max = -1.0, 1.0
            z_max = torch.amax(torch.abs(z), axis=(1,2), keepdims=True)  # farthest value from 0 in each channel
            scale = ((new_max - new_min) / (z_max*2))
            z *= scale

            # Calculate SNR metrics
            p_total_dbm = 10*torch.log10(torch.sum(10**(rx_pow_db.flatten(1)/10), 1))
            p_noise_total_dbm = (10*torch.log10((10**((rx_pow_db - snr)/10)).sum((1,2))))
            snr_total = p_total_dbm - p_noise_total_dbm
            
            if self.snr_inband:  # Apply in-band SNR calculation
                bw = 10*torch.log10((1/batch['T_s'])*(1+batch['beta']))
                snr -= bw[:,None,None]
                snr_total -= bw

            # Update batch with processed data
            batch['x'] = z
            batch['snr'] = snr.flatten(1)
            batch['snr_total'] = snr_total
            batch['pow_rx'] = rx_pow_db.flatten(1)
            return batch
        else:
            return batch
    
    def train_dataloader(self) -> DataLoader:
        """Return the training dataloader."""
        return self._data_loader(self.ds_train)

    def val_dataloader(self) -> DataLoader:
        """Return the validation dataloader."""
        return self._data_loader(self.ds_val)
    
    def test_dataloader(self) -> DataLoader:
        """Return the test dataloader."""
        return self._data_loader(self.ds_test)

    def _data_loader(self, dataset: Dataset) -> DataLoader:
        """
        Create a DataLoader with custom sampling for the TACM dataset.
        
        Args:
            dataset: The dataset to create a loader for
            
        Returns:
            Configured DataLoader with custom TACM sampler
        """
        gen = torch.Generator().manual_seed(self.seed)
        return DataLoader(
            dataset,
            batch_size=1,
            batch_sampler=TACMSampler(len(dataset), len(dataset.h_indices), self.batch_size, gen),
            shuffle=False,
            generator=gen,
            drop_last=False,
            pin_memory=True,
            num_workers=self.n_workers,
            persistent_workers=True,
        )
