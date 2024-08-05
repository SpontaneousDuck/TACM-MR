from typing import Optional
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

import numpy as np
import torch

from .MapGenerator import MapGenerator, TerrainType

class ScenarioGenerator:
    def __init__(
        self,
        n_receivers=1,
        n_transmitters=1,
        batch_size=1,
        h_receivers = 10,
        h_transmitters = 1.5,
        resolution=500,
        min_receiver_dist=200,
        min_path_distance=50,
        seed=42,
        n_workers=1,
        dtype=torch.float32,
        device: Optional[torch.device] = None
    ) -> None:
        self.map = []
        self.transmitters = []
        self.receivers = []
        self.n_rx = n_receivers
        self.n_tx = n_transmitters
        self.batch_size = batch_size
        self.h_tx = h_transmitters
        self.h_rx = h_receivers
        self.resolution = resolution
        self.min_receiver_dist = min_receiver_dist
        self.min_path_distance = min_path_distance
        self.seed=seed
        self.rng = torch.Generator(device=device).manual_seed(seed)
        self.device = device
        self._dtype = dtype
        self.map_gen = MapGenerator(resolution, n_workers=n_workers, seed=seed, dtype=dtype, device=device)

        self.RegenerateFullScenario()

    def RegenerateFullScenario(self):
        map_diagonal = np.sqrt(np.sum(np.power(self.resolution, 2)))
        if self.min_receiver_dist > map_diagonal:
            raise Exception("Invalid min receiver distance and map size specified")

        if self.min_path_distance > map_diagonal:
            raise Exception("Invalid min path distance and map size specified")
        
        self.map = self.map_gen() #Sample map generator

        self._create_nodes()
        return
    
    def _create_nodes(self):
        self.transmitters = torch.randint(self.resolution, size=(self.batch_size, self.n_tx, 3), generator=self.rng, dtype=self._dtype, device=self.device)
        self.transmitters[...,-1] = self.h_tx

        # Regenerate receivers until all meet the minimum distance requirement
        self.receivers = torch.randint(self.resolution, size=(self.batch_size, self.n_rx, 3), generator=self.rng, dtype=self._dtype, device=self.device)
        dist = torch.cdist(self.transmitters[...,:2], self.receivers[...,:2], 2)
        while dist.min() < self.min_receiver_dist:
            replace_idx = torch.any(dist < self.min_receiver_dist, dim=1)
            new_receivers = torch.randint(self.resolution, size=(self.batch_size, self.n_rx, 3), generator=self.rng, dtype=self._dtype, device=self.device)
            self.receivers[replace_idx] = new_receivers[replace_idx]
            dist = torch.cdist(self.transmitters[...,:2], self.receivers[...,:2], 2)
        self.receivers[...,-1] = self.h_rx

    def PlotMap(self, save_path=None):
        fig, axes = plt.subplots(nrows=1, figsize=(10, 10))
        im = axes.imshow(self.map.numpy(force=True), origin="lower", cmap="Blues")
        values = torch.unique(self.map.ravel()).numpy(force=True)

        for j in range(self.batch_size):
            for i in range(self.n_rx):
                axes.scatter(self.receivers[j, i,0].numpy(force=True), self.receivers[j, i,1].numpy(force=True), marker="o", color="g", s=50, zorder=10)

            for i in range(self.n_tx):
                axes.scatter(self.transmitters[j, i, 0].numpy(force=True), self.transmitters[j, i, 1].numpy(force=True), marker="o", color="y", s=50, zorder=10)

        # key_points = d["key_points"]
        # xtrans_coords, ytrans_coords = key_points[:, 0], key_points[:, 1]
        # axes.plot(xtrans_coords[::-1], ytrans_coords[::-1], "rx-", zorder=0)

        axes.axis("image")

        colors = [im.cmap(im.norm(value)) for value in values]
        # create a patch (proxy artist) for every color
        patches = [
            mpatches.Patch(
                color=colors[v_idx],
                label=f"{str(TerrainType(int(v))).split('.')[-1]} Terrain",
            )
            for v_idx, v in enumerate(values)
        ]
        patches.extend(
            [
                Line2D(
                    [],
                    [],
                    label="receiver",
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
        plt.grid(True)

        if save_path is not None:
            fig.savefig(save_path, bbox_inches="tight")
            return fig

        plt.show()
        return fig


if __name__ == "__main__":
    seed = 46

    scenarioGen = ScenarioGenerator(n_receivers=6, resolution=500, min_receiver_dist=1000, seed=seed)
    scenarioGen.PlotMap(save_path="Map.png")