from typing import Iterable, List, Iterator, Sequence, Tuple, Union, cast
import pytorch_lightning as pl
import h5py
import os
import torch
from torch.utils.data import Sampler
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split, StackDataset, Subset
# from scenario_generator import ScenarioGenerator
from sionna_torch.ApplyTimeChannel import ApplyTimeChannel


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
                 dataset_path: str, 
                 frame_size=1024, 
                 f_c=900e6, 
                 bw=30e3, 
                 reuse_factor=128,
                 gen_batch_size = 128,
                 pregen_channels=True, 
                 snr_inband=True, 
                 snr_range: List[int] = [-30,10],
                 seed: int = 42, 
                 n_rx = 1, 
                 device=None,
    ) -> None:
        self.map_res = 20 # meters/pixel
        self.map_size = 1000 # Pixels x Pixels map size
        self.gen_batch_size = gen_batch_size
        self.snr_inband = snr_inband
        self.device = device        

        print('Loading Data...')
        with h5py.File(dataset_path, "r") as f:
            x = torch.from_numpy(f['x'][()])
            y = torch.from_numpy(f['y'][()]).to(torch.long)
            h_t = torch.from_numpy(f['h_t'][()])
            # t0 = torch.from_numpy(f['T0'][()]).to(torch.long)
            beta = torch.from_numpy(f['beta'][()])
            T_s = torch.from_numpy(f['T_s'][()])
            S_idx = torch.from_numpy(f['S_idx'][()])

        print("Preprocessing Data")
        h_t = h_t.reshape(-1, n_rx, *h_t.shape[2:])
        # self.scenario_gen = ScenarioGenerator(n_receivers=n_rx, 
        #                                         batch_size=self.gen_batch_size, 
        #                                         map_size=self.map_size, 
        #                                         map_resolution=self.map_res, 
        #                                         min_receiver_dist=2, 
        #                                         max_iter=100, 
        #                                         frame_size=frame_size,
        #                                         f_c=f_c,
        #                                         bw=bw,
        #                                         seed=seed, 
        #                                         n_workers=1, 
        #                                         # target_total_p=True if scenario == "e" else False, 
        #                                         target_total_p=False, 
        #                                         dtype=x.dtype.to_real(),
        #                                         device=device)

        # self.snr_gen = torch.Generator(device).manual_seed(seed)
        # self.target_snr = torch.empty(n_rx, device=device)

        # for i in tqdm(range(0, len(x), self.terrain_reuse_factor), miniters=100, mininterval=10):
        #     self.target_snr = self.target_snr.uniform_(*snr_range, generator=self.snr_gen)
        #     # snr1 = torch.empty(self.n_rx, device=self.trainer.strategy.root_device).uniform_(self.snr_min, self.snr_max, generator=self.snr_gen)
        #     # snr2 = torch.empty(self.n_rx, device=self.trainer.strategy.root_device).uniform_(self.snr_min, -10, generator=self.snr_gen)
        #     # snr3 = torch.empty(self.n_rx, device=self.trainer.strategy.root_device).uniform_(self.snr_min, -10, generator=self.snr_gen)
        #     # self.target_snr = torch.where(torch.randint(0,2,self.n_rx, generator=self.snr_gen, dtype=torch.bool), snr1, snr2)
        #     # self.target_snr = torch.where(torch.randint(0,1,self.n_rx, generator=self.snr_gen), self.target_snr, snr3)
            
        #     target_power = self.target_snr + self.scenario_gen.chan_gen.get_noise_power()
        #     h_t[i] = self.scenario_gen.RegenerateFullScenario(target_power)

        self.data_x = x
        self.data_y = y
        self.data_h_t = h_t
        self.data_T_s = T_s
        self.data_beta = beta
        self.data_S_idx = S_idx

    def __getitem__(self, index):
        # return {x=x[index], y=y, T_s=T_s, beta=beta, S_idx=sym_idx, snr=snr, snr_total=total_snr, pow_rx=pow_rx}
        x = self.data_x[index[0]]
        h_t = self.data_h_t[index[1]]
        return {'x':x, 'h_t': h_t, 'y':self.data_y[index[0]], 'T_s':self.data_T_s[index[0]], 'beta':self.data_beta[index[0]], 'S_idx':self.data_S_idx[index[0]]}
        # z, rx_pow_db, snr = self._apply_channel(x[:,:1,None].to(self.device), h_t.to(self.device), None)
        # z = z.squeeze(2,3)[...,self.l_min*-1:self.l_max*-1]

        # # Per-frame normalize to -1.0:1.0
        # new_min, new_max = -1.0, 1.0
        # z_max = torch.amax(torch.abs(z), axis=(1,2), keepdims=True) # farthest value from 0 in each channel
        # scale = ((new_max - new_min) / (z_max*2))
        # z *= scale

        # p_total_dbm = 10*torch.log10(torch.sum(10**(rx_pow_db.flatten(1)/10), 1))
        # p_noise_total_dbm = (10*torch.log10((10**((rx_pow_db - snr)/10)).sum((1,2))))
        # snr_total = p_total_dbm - p_noise_total_dbm
        # if self.snr_inband: # inband snr
        #     bw = 10*torch.log10((1/self.data_T_s[index])*(1+self.data_beta[index]))
        #     snr -= bw[:,None,None]
        #     snr_total -= bw

        # return {'x':x, 'h_t': h_t, 'y':self.data_y[index], 'T_s':self.data_T_s[index], 'beta':self.data_beta[index], 'S_idx':self.data_S_idx[index], 'snr': snr.flatten(1), 'snr_total': snr_total, 'pow_rx': rx_pow_db.flatten(1)}

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

class TACM2024DataModule(pl.LightningDataModule):
    def __init__(self, 
                 dataset_path: str, 
                 batch_size, 
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
        self.dataset_path = os.path.expanduser(dataset_path)
        self.batch_size = batch_size if scenario != "c" else batch_size//n_rx
        self.terrain_reuse_factor = reuse_factor
        self.gen_batch_size = gen_batch_size
        self.frame_size = frame_size
        self.transforms = []
        self.seed = seed
        self.n_rx = n_rx
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
        pass

    def setup(self, stage: str = None):
        if not len(self.ds_train) or not len(self.ds_val) or not len(self.ds_test):
            self.generator = torch.Generator().manual_seed(self.seed)
            
            self.snr_gen = torch.Generator(self.trainer.strategy.root_device).manual_seed(self.seed)
            self.l_min, self.l_max = -6, int(np.ceil(3e-6*self.bw)) + 6
            l_tot = self.l_max-self.l_min+1
            self._apply_channel = ApplyTimeChannel(self.frame_size, l_tot=l_tot, rng=self.snr_gen, add_awgn=True, device=self.trainer.strategy.root_device)

            self.ds = TACMDataset(self.dataset_path,
                                  frame_size=self.frame_size,
                                  f_c = self.f_c,
                                  bw = self.bw,
                                  snr_inband = self.snr_inband,
                                  snr_range = self.snr_range,
                                  seed = self.seed,
                                  n_rx = self.n_rx,
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
        return self._data_loader(self.ds_train, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._data_loader(self.ds_val, shuffle=False)
    
    def test_dataloader(self) -> DataLoader:
        return self._data_loader(self.ds_test, shuffle=False)

    def _data_loader(self, dataset: Dataset, shuffle: bool = False) -> DataLoader:
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
