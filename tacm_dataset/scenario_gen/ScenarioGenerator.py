from typing import Optional
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

import numpy as np
import torch
import itertools

from .MapGenerator import MapGenerator, TerrainType
from .ChannelGenerator import ChannelGenerator


class ScenarioGenerator:
    def __init__(
        self,
        n_receivers=1,
        n_transmitters=1,
        batch_size=1,
        h_receivers = 10,
        h_transmitters = 1.5,
        map_size=500,
        map_resolution=12,
        min_receiver_dist=200,
        frame_size=1024,
        f_c: float = .92e9,
        bw: float = 30e3,
        noise_power_dB = None,
        seed=42,
        n_workers=1,
        noise_type='perlin',
        max_iter=5,
        optimizer_error = 0.5,
        target_total_p = True,
        dtype=torch.float32,
        device: Optional[torch.device] = None,
        debug = False
    ) -> None:
        self.map = []
        self.transmitters = []
        self.receivers = []
        self.n_rx = n_receivers
        self.n_tx = n_transmitters
        self.batch_size = batch_size
        self.h_tx = h_transmitters
        self.h_rx = h_receivers
        self.map_size = map_size
        self.map_resolution = map_resolution
        self.min_receiver_dist = min_receiver_dist
        self.seed=seed
        self.rng = torch.Generator(device=device).manual_seed(seed)
        self.device = device
        self._dtype = dtype
        self._debug = debug
        self.max_iter = max_iter
        self.target_total_p = target_total_p
        self.map_gen = MapGenerator(map_size, noise_type=noise_type, n_workers=n_workers, seed=seed, dtype=dtype, device=device)

        # data type
        assert not dtype.is_complex, "'dtype' must be complex type"
        self._dtype_complex = dtype.to_complex()

        self.chan_gen = ChannelGenerator(
            map_resolution = map_resolution,
            n_rx = n_receivers,
            n_tx = n_transmitters,
            frame_size = frame_size,
            f_c = f_c,
            bw = bw,
            noise_power_dB = noise_power_dB,
            direction = "uplink",
            target_total_p = target_total_p,
            minDist = min_receiver_dist,
            batch_size = batch_size,
            max_iters = 3,
            max_error = optimizer_error,
            seed = seed,
            dtype = dtype,
            device = device,
            debug = debug,

        )

    def RegenerateFullScenario(self, target_pow_db=None):
        map_diagonal = np.sqrt(np.sum(np.power(self.map_size, 2)))
        if self.min_receiver_dist > map_diagonal:
            raise Exception("Invalid min receiver distance and map size specified")
        
        self.map = self.map_gen() #Sample map generator
        
        self._create_nodes()
        
        if target_pow_db is not None:
            iter_control = 0
            target_pow_db = target_pow_db.to(self.device)
            while(True):
                iter_control += 1
                try:
                    h_T, self.receivers = self.chan_gen(self.map, self.transmitters, self.receivers, target_pow_db, los_requested=False)
                    return h_T
                except AssertionError as e:
                    if self._debug: print(e)
                    self._create_nodes()
                
                if iter_control > self.max_iter:
                    raise Exception("Impossible problem found. Something is broken")
        else:
            self.chan_gen.sionna.update_topology(self.transmitters.repeat((128,1,1)), self.receivers.repeat((128,1,1)), self.map, map_resolution=self.map_resolution, direction='uplink', los_requested=False)
            z, _ = self.chan_gen.sionna.generate_channels()

            return z.to(self.device)[:1]
    
    def _create_nodes(self):
        self.transmitters = torch.randint(self.map_size, size=(1, self.n_tx, 3), generator=self.rng, dtype=self._dtype, device=self.device)
        self.transmitters[...,-1] = self.h_tx

        # Regenerate receivers until all meet the minimum distance requirement
        self.receivers = torch.randint(self.map_size, size=(1, self.n_rx, 3), generator=self.rng, dtype=self._dtype, device=self.device)
        dist = torch.cdist(self.transmitters[...,:2], self.receivers[...,:2], 2)
        while dist.min() < self.min_receiver_dist:
            replace_idx = torch.any(dist < self.min_receiver_dist, dim=1)
            new_receivers = torch.randint(self.map_size, size=(1, self.n_rx, 3), generator=self.rng, dtype=self._dtype, device=self.device)
            self.receivers[replace_idx] = new_receivers[replace_idx]
            dist = torch.cdist(self.transmitters[...,:2], self.receivers[...,:2], 2)
        self.receivers[...,-1] = self.h_rx

    def PlotMap(self, SNRs=None, save_path=None):
        fig, axes = plt.subplots(nrows=1, figsize=(4, 4))
        extent = (0, self.map_size*self.map_resolution, 0, self.map_size*self.map_resolution)
        im = axes.imshow(self.map.numpy(force=True), extent=extent, origin="lower", cmap="Blues")
        values = torch.unique(self.map.ravel()).numpy(force=True)

        for j in range(1):
            for i in range(self.n_rx):
                x = self.receivers[j, i,0].numpy(force=True) * self.map_resolution
                y = self.receivers[j, i,1].numpy(force=True) * self.map_resolution
                axes.scatter(x, y, marker="o", color="g", s=50, zorder=10)

            for i in range(self.n_tx):
                x = self.transmitters[j, i, 0].numpy(force=True) * self.map_resolution
                y = self.transmitters[j, i, 1].numpy(force=True) * self.map_resolution
                axes.scatter(x, y, marker="o", color="y", s=50, zorder=10)

            prop_paths = np.array(list(itertools.product(self.receivers[j,:,:2].numpy(force=True)*self.map_resolution, self.transmitters[j,:,:2].numpy(force=True)*self.map_resolution)))
            xtrans_coords, ytrans_coords = prop_paths[..., 0].T, prop_paths[..., 1].T
            axes.plot(xtrans_coords, ytrans_coords, "r.-", zorder=0)

        axes.axis("image")

        colors = [im.cmap(im.norm(value)) for value in values]
        # create a patch (proxy artist) for every color
        patches = [
            mpatches.Patch(
                color=colors[v_idx],
                label=f"{str(TerrainType(int(v)).name)} Terrain",
            )
            for v_idx, v in enumerate(values)
        ]
        patches.extend(
            [
                Line2D(
                    [],
                    [],
                    label="Receiver",
                    color="white",
                    marker="o",
                    markersize=12,
                    markerfacecolor="g",
                )
            ]
        )
        patches.extend(
            [
                Line2D(
                    [],
                    [],
                    label="Sender",
                    color="white",
                    marker="o",
                    markersize=12,
                    markerfacecolor="y",
                )
            ]
        )
        # put those patched as legend-handles into the legend
        plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
        plt.legend(handles=patches, loc='upper right', borderaxespad=0.0)
        plt.xlabel("Meters")
        plt.ylabel("Meters")
        plt.tight_layout()
        plt.grid(True)

        if save_path is not None:
            fig.savefig(save_path, bbox_inches="tight")
            return fig

        plt.show()
        return fig


if __name__ == "__main__":
    seed = 46

    scenarioGen = ScenarioGenerator(n_receivers=6, resolution=500, min_receiver_dist=1000, seed=seed)
    scenarioGen.PlotMap(save_path="Map.svg")