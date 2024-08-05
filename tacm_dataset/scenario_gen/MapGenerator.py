from enum import IntEnum
import torch
import pyfastnoisesimd

DEFAULT_MAP_CONFIG = {
    "foliage_freq": 0.005,
    "foliage_octaves": 3,
    "foliage_octaves_factor": 7.1,
    "foliage_levels": 10,
    "foliage_redistribution": 0.7,
    "pop_octaves": 3,
    "pop_freq": 0.0001,
    "pop_octaves_factor": 5,
    "pop_levels": 11,
    "pop_redistribution": 4
}

class TerrainType(IntEnum):
    Urban = 1
    Rural = 0

class MapGenerator():
    def __init__(self,
                  width: int,
                  freq=DEFAULT_MAP_CONFIG["pop_freq"],
                  octaves=DEFAULT_MAP_CONFIG["pop_octaves"],
                  octaves_factor=DEFAULT_MAP_CONFIG["pop_octaves_factor"],
                  levels=DEFAULT_MAP_CONFIG["pop_levels"],
                  redistribution=DEFAULT_MAP_CONFIG["pop_redistribution"],
                  island=True,
                  urban_threshold=0.5,
                  n_workers=1,
                  seed=42,
                  dtype=torch.float32,
                  device=None) -> None:
        self.width = width
        self.freq = freq
        self.octaves = octaves
        self.octaves_factor = octaves_factor
        self.levels = levels
        self.redistribution = redistribution
        self.island = island
        self.urban_threshold = urban_threshold

        self.fns = pyfastnoisesimd.Noise(seed=seed, numWorkers=n_workers)
        self.fns.noiseType = pyfastnoisesimd.NoiseType.Simplex

        self.dtype = dtype
        self.device = device

    def __call__(self):
        # Initialize map
        z = torch.zeros((self.width, self.width), dtype=self.dtype, device=self.device)

        # Apply noise maps with different frequencies
        of = 1
        for _ in range(self.octaves):
            self.fns.seed = self.fns.seed + 1
            self.fns.frequency = self.freq * of
            z += (1 / of) * torch.from_numpy(self.fns.genAsGrid([self.width, self.width])).type(self.dtype).to(self.device)
            of *= self.octaves_factor

        # Enable population islands
        znorm = (2 * (z - z.min()) / (z.max() - z.min())) - 1  # norm from -1 to 1
        _b = torch.linspace(-1, 1, self.width, dtype=self.dtype, device=self.device)
        _x, _y = torch.meshgrid(_b, _b, indexing='xy')
        d = -(1 - ((1 - _x**2) * (1 - _y**2)))
        if self.island:
            d *= -1
        z = (znorm + (1 - d)) / 2

        # Control diff between peaks and troughs
        z = (z - z.min()) / (z.max() - z.min())  # norm from 0 to 1
        z = torch.pow(z, self.redistribution)

        # Quantize to specific terrain levels
        z = torch.bucketize(z, torch.linspace(z.min(), z.max() + 0.0000001, self.levels + 1, dtype=self.dtype, device=self.device))
        z = (z - z.min()) / (z.max() - z.min())  # norm from 0 to 1

        # Resolve map to classes
        return torch.where(z >= self.urban_threshold, TerrainType.Urban, TerrainType.Rural)
