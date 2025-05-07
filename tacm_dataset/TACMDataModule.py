from typing import Iterable, List, Iterator, Sequence, Tuple, Union, cast
import pytorch_lightning as pl
import h5py
import os
import hashlib
import urllib
import zipfile
import glob
import torch
import pandas as pd
from pathlib import Path
from torch.utils.data import Sampler
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split, StackDataset, Subset
# from scenario_generator import ScenarioGenerator
from .sionna_torch.ApplyTimeChannel import ApplyTimeChannel
from .scenario_gen import ScenarioGenerator


# TODO: subsample factor
# TODO: download dataset from Chad's site

class TACMDataset:
    r"""Dataset as a stacking of multiple datasets.

    This class is useful to assemble different parts of complex input data, given as datasets.

    Example:
        >>> # xdoctest: +SKIP
        >>> images = ImageDataset()
        >>> texts = TextDataset()
        >>> tuple_stack = StackDataset(images, texts)
        >>> tuple_stack[0] == (images[0], texts[0])
        >>> dict_stack = StackDataset(image=images, text=texts)
        >>> dict_stack[0] == {'image': images[0], 'text': texts[0]}

    Args:
        *args (Dataset): Datasets for stacking returned as tuple.
        **kwargs (Dataset): Datasets for stacking returned as dict.
    """

    def __init__(self, 
                 dataset_root: str, 
                 frame_size=1024, 
                 chan_file: str = None,
                 scenario: str = 'a',
                 n_rx = 1, 
                 map_res = 20,
                 device=None,
    ) -> None:

        print('Loading Data...')
        with h5py.File(os.path.join(dataset_root, "CSPB.ML.2018R2_nonoise.hdf5"), "r") as f:
            x = torch.from_numpy(f['x'][()])
            y = torch.from_numpy(f['y'][()]).to(torch.long)
            t0 = torch.from_numpy(f['T0'][()]).to(torch.long)
            beta = torch.from_numpy(f['beta'][()])
            T_s = torch.from_numpy(f['T_s'][()])
            
        print("Preprocessing Data")
        S_idx = torch.linspace(0, 1.0, x.shape[-1]).repeat(x.shape[0], 1)
        S_idx *= (32768/T_s[:,None])
        S_idx = S_idx.floor_().short()

        # Reshape data to frame_size
        # Need to be careful to use even multiples though so we don't break original fram barriers (32768 samples)
        if 32768 % frame_size:
            raise RuntimeError("frame_size is not an even divisor of the original frame size 32768.")
        x = x.reshape((-1, 1, frame_size))
        reshape_factor = 32768//frame_size
        y = torch.repeat_interleave(y, reshape_factor)
        t0 = torch.repeat_interleave(t0, reshape_factor)
        beta = torch.repeat_interleave(beta, reshape_factor)
        T_s = torch.repeat_interleave(T_s, reshape_factor)
        S_idx = S_idx.reshape((-1, frame_size))
        S_idx -= S_idx[:,:1]

        # Scale signal to 0dB power
        transmit_power = 0 # 0dB
        signal_power = torch.mean(torch.abs(x) ** 2, -1)[...,None]
        target_snr_linear = 10 ** (transmit_power / 10)
        # occupied_bw = 1 / np.repeat(t0[i:i+step], x1.shape[0]/step)[:,None,None,None]
        # signal_scale_linear = np.sqrt((target_snr_linear * occupied_bw) / signal_power)
        signal_scale_linear = torch.sqrt((target_snr_linear) / signal_power)
        x = x * signal_scale_linear
        
        hdf5_path = chan_file if chan_file is not None else os.path.join(dataset_root, f"TACM_2025_1_channels_{scenario}_{n_rx if scenario == 'd' else 8}rx.h5py")
        if not os.path.exists(hdf5_path):
            raise FileNotFoundError(f"Channel file {hdf5_path} does not exist, force channel generation.")
        with h5py.File(hdf5_path, "r") as f:
            h_t = torch.from_numpy(f['h_t'][()])
            tx_xy = torch.from_numpy(f['tx_xy'][()])
            rx_xy = torch.from_numpy(f['rx_xy'][()])
            topography = torch.from_numpy(f['topography'][()]) 

        h_t = h_t[:,:n_rx]
        tx_xy = tx_xy[:,:n_rx]
        tx_xy[..., :2] *= map_res
        rx_xy = rx_xy[:,:n_rx]
        rx_xy[..., :2] *= map_res

        self.data_x = x
        self.data_y = y
        self.data_T_s = T_s
        self.data_beta = beta
        self.data_S_idx = S_idx
        match scenario:
            case "a" | "d":
                self.data_h_t = h_t
                self.data_tx_xy = tx_xy
                self.data_rx_xy = rx_xy
                self.topography = topography
            case "c":
                self.data_h_t = h_t[:, :n_rx].flatten(0,1).unsqueeze(1)
                self.data_tx_xy = tx_xy[:].repeat(n_rx, 1, 1)
                self.data_rx_xy = rx_xy[:, :n_rx].flatten(0,1).unsqueeze(1)
                self.topography = topography.repeat(n_rx, 1, 1)
            case "b":
                self.data_x = x[::n_rx]
                self.data_y = y[::n_rx]
                self.data_T_s = T_s[::n_rx]
                self.data_beta = beta[::n_rx]
                self.data_S_idx = S_idx[::n_rx]
                self.data_h_t = h_t[::n_rx, :n_rx]
                self.data_tx_xy = tx_xy[::n_rx]
                self.data_rx_xy = rx_xy[::n_rx, :n_rx]
                self.topography = topography[::n_rx]
            case _:
                raise ValueError("Unknown data scenario")

    def __getitem__(self, index):
        # return {x=x[index], y=y, T_s=T_s, beta=beta, S_idx=sym_idx, snr=snr, snr_total=total_snr, pow_rx=pow_rx}
        x = self.data_x[index[0]]
        h_t = self.data_h_t[index[1]]
        return {'x':x, 'h_t': h_t, 'y':self.data_y[index[0]], 'T_s':self.data_T_s[index[0]], 'beta':self.data_beta[index[0]], 'S_idx':self.data_S_idx[index[0]], 'tx_xy':self.data_tx_xy[index[1]], 'rx_xy':self.data_rx_xy[index[1]]}

    # def __getitems__(self, indices: list):
    #     # add batched sampling support when parent datasets supports it.
    #     if isinstance(self.datasets, dict):
    #         dict_batch: List[T_dict] = [{} for _ in indices]
    #         for k, dataset in self.datasets.items():
    #             if callable(getattr(dataset, "__getitems__", None)):
    #                 items = dataset.__getitems__(indices)  # type: ignore[attr-defined]
    #                 if len(items) != len(indices):
    #                     raise ValueError(
    #                         "Nested dataset's output size mismatch."
    #                         f" Expected {len(indices)}, got {len(items)}"
    #                     )
    #                 for data, d_sample in zip(items, dict_batch):
    #                     d_sample[k] = data
    #             else:
    #                 for idx, d_sample in zip(indices, dict_batch):
    #                     d_sample[k] = dataset[idx]
    #         return dict_batch

    def __len__(self):
        return len(self.data_x)

class TACMSampler(Sampler[List[int]]):
    r"""Returns a mixed batch of samples and channels.

    Args:
        sampler (Sampler or Iterable): Base sampler. Can be any iterable object
        batch_size (int): Size of mini-batch.
    """

    def __init__(self, x_len, h_len, batch_size: int, generator=None) -> None:
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
        x_iter = torch.randperm(self.x_len, generator=self.generator).tolist()
        h_perm = torch.randperm(self.h_len, generator=self.generator)
        repeat_factor = self.x_len // self.h_len
        h_perm = h_perm.repeat(repeat_factor).tolist()

        yield from zip(x_iter, h_perm)

    def __iter__(self) -> Iterator[List[int]]:
        batch = [(0,0)] * self.batch_size
        idx_in_batch = 0
        for idx in self._sampler_iter():
            batch[idx_in_batch] = idx
            idx_in_batch += 1
            if idx_in_batch == self.batch_size:
                yield batch
                idx_in_batch = 0
                batch = [0] * self.batch_size
        if idx_in_batch > 0:
            yield batch[:idx_in_batch]

    def __len__(self) -> int:
        return (self.x_len + self.batch_size - 1) // self.batch_size

class TACMSubset(Dataset):
    r"""
    Subset of a dataset at specified indices.

    Args:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """

    dataset: TACMDataset
    x_indices: Sequence[int]
    h_indices: Sequence[int]

    def __init__(self, dataset: TACMDataset, x_indices: Sequence[int], h_indices: Sequence[int]) -> None:
        self.dataset = dataset
        self.x_indices = x_indices
        self.h_indices = h_indices

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return self.dataset[[(self.x_indices[x_i], self.h_indices[h_i]) for x_i, h_i in idx]]
        return self.dataset[(self.x_indices[idx[0]], self.h_indices[idx[1]])]

    # def __getitems__(self, indices: List[int]) -> List[T_co]:
    #     # add batched sampling support when parent dataset supports it.
    #     # see torch.utils.data._utils.fetch._MapDatasetFetcher
    #     if callable(getattr(self.dataset, "__getitems__", None)):
    #         return self.dataset.__getitems__([self.indices[idx] for idx in indices])  # type: ignore[attr-defined]
    #     else:
    #         return [self.dataset[self.indices[idx]] for idx in indices]

    def __len__(self):
        return len(self.x_indices)

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
    def __init__(self, 
                 dataset_root: str, 
                 batch_size, 
                 download: bool = False,
                 gen_chan: bool = False,
                 n_topo: int = 28000,
                 in_memory: bool = True,
                 frame_size=1024, 
                 f_c=900e6, 
                 bw=30e3, 
                 reuse_factor=128,
                 gen_batch_size = 128,
                 snr_inband=True, 
                 snr_range: List[int] = [-30,10],
                 scenario=None, 
                 seed: int = 42, 
                 n_rx = 1, 
                 n_workers:int = 8):
        super().__init__()
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
        self.map_res = 20 # meters/pixel
        self.map_size = 1000 # Pixels x Pixels map size
        self.n_workers = n_workers
        self.snr_inband = snr_inband
        self.snr_range = snr_range
        self.scenario = scenario

        noise_power_db = -173.8 + 10 * np.log10(bw)
        self.noise_power_lin = 10**((noise_power_db)/10)
       
        self.classes = ['bpsk', 'qpsk', '8psk', 'dqpsk', 'msk', '16qam', '64qam', '256qam']
        self.ds_train, self.ds_val, self.ds_test = [], [], []


    def prepare_data(self): 
        if self.download or not os.path.exists(os.path.join(self.dataset_root, "CSPB.ML.2018R2_nonoise.hdf5")):
            print("Downloading CSPB files")
            download_root = os.path.join(self.dataset_root, "CSPB_ML_2018R2_nonoise_zips")
            os.makedirs(download_root, exist_ok=True)
            extract_root = os.path.join(self.dataset_root, "CSPB_ML_2018R2_nonoise_signals")
            os.makedirs(extract_root, exist_ok=True)

            for url, zip_md5 in CSPB2018R2_NONOISE_META:
                fname = os.path.basename(url)
                if 'zip' in os.path.basename(url):
                    fpath = os.path.join(download_root, fname)
                else: 
                    fpath = os.path.join(extract_root, fname)

                # Download file
                print("Downloading " + url + " to " + fpath)
                with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "pytorch/vision"})) as response:
                    content = iter(lambda: response.read(1024*32), b"")
                    with open(fpath, "wb") as fh:
                        for chunk in content:
                            if not chunk:
                                continue
                            fh.write(chunk)

                # check integrity of downloaded file
                md5_alg = hashlib.md5(usedforsecurity=False)
                with open(fpath, "rb") as f:
                    for chunk in iter(lambda: f.read(1024*1024), b""):
                        md5_alg.update(chunk)
                if zip_md5 != md5_alg.hexdigest():
                    raise RuntimeError(f"File not found or corrupted: {fpath}")

                # Unzip zip file downloaded
                if 'zip' in fname:
                    print(f"Extracting {fpath} to {extract_root}")
                    with zipfile.ZipFile(fpath, "r", compression=zipfile.ZIP_STORED) as zipf:
                        zipf.extractall(extract_root)

            fnames = glob.glob(os.path.join(extract_root, "*", "signal*.tim"))
            fnames = sorted(fnames, key=lambda x: int(Path(x).stem.split("_")[-1]))

            x = np.array([np.fromfile(fname)[1:].view(np.complex64) for fname in fnames])

            labels = pd.read_csv(os.path.join(extract_root, "signal_record_C_2023.txt"), header=None, sep='\s+', index_col=0, names=['label', 'T0', 'cfo', 'beta', 'U', 'D', 'SNR', 'P(N)'])
            y = np.array([self.classes.index(label) for label in labels.label])

            # Set all MSK to a beta of 0.5, this gives us about null-to-null for bandwidth
            labels.loc[labels['label'] == "msk", 'beta'] = 0.5

            t0 = labels.T0.to_numpy()
            d = labels.D.to_numpy()
            u = labels.U.to_numpy()

            with h5py.File(os.path.join(self.dataset_root, "CSPB.ML.2018R2_nonoise.hdf5"), "w") as f:
                f['x'] = x
                f['y'] = y
                f['T0'] = t0
                f['cfo'] = labels.cfo.to_numpy()
                f['beta'] = labels.beta.to_numpy()
                f['U'] = u
                f['D'] = d
                f['T_s'] = 1 / ((1/t0)*(d/u))

        h5py_path = os.path.join(self.dataset_root, f"TACM_2025_1_channels_{self.scenario}_{self.n_rx if self.scenario == 'd' else 8}rx.h5py")
        if self.gen_chan or not os.path.exists(h5py_path):
            print("Generating Channels File")
            batch_size = 128
            n_rx = self.n_rx if self.scenario == "d" else 8
            dev = self.trainer.strategy.root_device if self.trainer is not None else torch.get_default_device()
            scenario_gen = ScenarioGenerator(n_receivers=n_rx, 
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
                                            device=dev)

            h_t = torch.empty((self.n_topo, n_rx, 1, 1, 1, 1, 14), device=torch.device('cpu'), dtype=torch.complex64)
            topography = torch.empty(self.n_topo, self.map_size, self.map_size, device=torch.device('cpu'))
            tx_xy = torch.empty(self.n_topo, 1, 3, device=torch.device('cpu'))
            rx_xy = torch.empty(self.n_topo, n_rx, 3, device=torch.device('cpu'))
            snr_gen = torch.Generator().manual_seed(self.seed)
            snr_min = self.snr_range[0]
            snr_max = self.snr_range[1]

            target_snr = torch.empty(n_rx)
            for i in tqdm(range(0, self.n_topo), miniters=100, mininterval=10):

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

            print("Writing: ", h5py_path)

            with h5py.File(h5py_path, "w") as f:
                f.create_dataset('h_t', data=h_t.numpy(force=True), compression="gzip")
                f.create_dataset('tx_xy', data=tx_xy, compression="gzip")
                f.create_dataset('rx_xy', data=rx_xy, compression="gzip")
                f.create_dataset('topography', data=topography, compression="gzip")

    def setup(self, stage: str = None):
        if not len(self.ds_train) or not len(self.ds_val) or not len(self.ds_test):
            self.generator = torch.Generator().manual_seed(self.seed)
            
            self.l_min, self.l_max = -6, int(np.ceil(3e-6*self.bw)) + 6
            l_tot = self.l_max-self.l_min+1
            dev = self.trainer.strategy.root_device if self.trainer is not None else torch.get_default_device()
            self.snr_gen = torch.Generator(dev).manual_seed(self.seed)
            self._apply_channel = ApplyTimeChannel(self.frame_size, l_tot=l_tot, rng=self.snr_gen, add_awgn=True, device=dev)

            self.ds = TACMDataset(self.dataset_root,
                                  frame_size=self.frame_size,
                                  n_rx = self.n_rx,
                                  scenario = self.scenario,
                                  map_res = self.map_res,
                                  )

            subset_lengths = [0.6, 0.2, 0.2]
            subset_lengths_x = [int(len(self.ds.data_x) * frac) for frac in subset_lengths]
            remainder = len(self.ds) - sum(subset_lengths_x)
            # add 1 to all the lengths in round-robin fashion until the remainder is 0
            for i in range(remainder):
                idx_to_add_at = i % len(subset_lengths_x)
                subset_lengths_x[idx_to_add_at] += 1
            indices_x = torch.randperm(sum(subset_lengths_x), generator=self.generator).tolist()  # type: ignore[arg-type, call-overload]

            subset_lengths_h = [int(len(self.ds.data_h_t) * frac) for frac in subset_lengths]
            remainder = len(self.ds.data_h_t) - sum(subset_lengths_h)
            # add 1 to all the lengths in round-robin fashion until the remainder is 0
            for i in range(remainder):
                idx_to_add_at = i % len(subset_lengths_h)
                subset_lengths_h[idx_to_add_at] += 1
            indices_h = torch.randperm(sum(subset_lengths_h), generator=self.generator).tolist()  # type: ignore[arg-type, call-overload]

            self.ds_train, self.ds_val, self.ds_test = [
                TACMSubset(self.ds, indices_x[offset_x - length_x : offset_x], indices_h[offset_h - length_h : offset_h])
                for offset_x, length_x, offset_h, length_h 
                in zip(np.cumsum(subset_lengths_x).tolist(), subset_lengths_x, np.cumsum(subset_lengths_h).tolist(), subset_lengths_h)
            ]

    def on_after_batch_transfer(self, batch, dataloader_idx):
        if isinstance(batch, dict):
            x = batch['x']
            h_t = batch['h_t']

            z, rx_pow_db, snr = self._apply_channel(x[:,:1,None], h_t, self.noise_power_lin, None)
            z = z.squeeze(2,3)[...,self.l_min*-1:self.l_max*-1]

            # Per-frame normalize to -1.0:1.0
            new_min, new_max = -1.0, 1.0
            z_max = torch.amax(torch.abs(z), axis=(1,2), keepdims=True) # farthest value from 0 in each channel
            scale = ((new_max - new_min) / (z_max*2))
            z *= scale

            p_total_dbm = 10*torch.log10(torch.sum(10**(rx_pow_db.flatten(1)/10), 1))
            p_noise_total_dbm = (10*torch.log10((10**((rx_pow_db - snr)/10)).sum((1,2))))
            snr_total = p_total_dbm - p_noise_total_dbm
            if self.snr_inband: # inband snr
                bw = 10*torch.log10((1/batch['T_s'])*(1+batch['beta']))
                snr -= bw[:,None,None]
                snr_total -= bw

            batch['x'] = z
            batch['snr'] = snr.flatten(1)
            batch['snr_total'] = snr_total
            batch['pow_rx'] = rx_pow_db.flatten(1)
            return batch
        else:
            return batch
    
    def train_dataloader(self) -> DataLoader:
        return self._data_loader(self.ds_train)

    def val_dataloader(self) -> DataLoader:
        return self._data_loader(self.ds_val)
    
    def test_dataloader(self) -> DataLoader:
        return self._data_loader(self.ds_test)

    def _data_loader(self, dataset: Dataset) -> DataLoader:
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
