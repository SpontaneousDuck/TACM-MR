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

class FractalPerlin2D(object):
    def __init__(self, shape, resolutions, factors, generator=torch.random.default_generator):
        shape = shape if len(shape)==3 else (None,)+shape
        self.shape = shape
        self.factors = factors
        self.generator = generator
        self.device = generator.device
        self.resolutions = resolutions
        self.grid_shapes = [(shape[1]//res[0], shape[2]//res[1]) for res in resolutions]
        
        #precomputed tensors
        self.linxs = [torch.linspace(0,1,gs[1],device=self.device) for gs in self.grid_shapes]
        self.linys = [torch.linspace(0,1,gs[0],device=self.device) for gs in self.grid_shapes]
        self.tl_masks = [self.fade(lx)[None,:]*self.fade(ly)[:,None] for lx, ly in zip(self.linxs, self.linys)]
        self.tr_masks = [torch.flip(tl_mask,dims=[1]) for tl_mask in self.tl_masks]
        self.bl_masks = [torch.flip(tl_mask,dims=[0]) for tl_mask in self.tl_masks]
        self.br_masks = [torch.flip(tl_mask,dims=[0,1]) for tl_mask in self.tl_masks]
        
    def fade(self, t):
        return 6 * t**5 - 15 * t**4 + 10 * t**3
    
    def perlin_noise(self, octave, batch_size):
        res = self.resolutions[octave]
        angles = torch.zeros((batch_size, res[0]+2, res[1]+2), device=self.device)
        angles.uniform_(0, 6.28318530718, generator=self.generator) # 0:2pi
        rx = torch.cos(angles)[:,:,:,None]*self.linxs[octave]
        ry = torch.sin(angles)[:,:,:,None]*self.linys[octave]
        prx, pry = rx[:,:,:,None,:], ry[:,:,:,:,None]
        nrx, nry = -torch.flip(prx, dims=[4]), -torch.flip(pry, dims=[3])
        br = prx[:,:-1,:-1] + pry[:,:-1,:-1]
        bl = nrx[:,:-1,1:] + pry[:,:-1,1:]
        tr = prx[:,1:,:-1] + nry[:,1:,:-1]
        tl = nrx[:,1:,1:] + nry[:,1:,1:]
        
        grid_shape = self.grid_shapes[octave]
        grids = self.br_masks[octave]*br + self.bl_masks[octave]*bl + self.tr_masks[octave]*tr + self.tl_masks[octave]*tl
        noise = grids.permute(0,1,3,2,4).reshape((batch_size, self.shape[1]+grid_shape[0], self.shape[2]+grid_shape[1]))

        A = torch.randint(0,grid_shape[0],(batch_size,), device=self.device, generator=self.generator)
        B = torch.randint(0,grid_shape[1],(batch_size,), device=self.device, generator=self.generator)
        noise = torch.stack([noise[n,a:a-grid_shape[0], b:b-grid_shape[1]] for n,(a,b) in enumerate(zip(A,B))])
        return noise
    
    def __call__(self, batch_size=None):
        batch_size = self.shape[0] if batch_size is None else batch_size
        shape = (batch_size,) + self.shape[1:]
        noise = torch.zeros(shape, device=self.device)
        for octave, factor in enumerate(self.factors):
            noise += factor*self.perlin_noise(octave, batch_size=batch_size)
        return noise

class MapGenerator():
    def __init__(self,
                  width: int,
                  freq=DEFAULT_MAP_CONFIG["pop_freq"],
                  octaves=DEFAULT_MAP_CONFIG["pop_octaves"],
                  octaves_factor=DEFAULT_MAP_CONFIG["pop_octaves_factor"],
                  levels=DEFAULT_MAP_CONFIG["pop_levels"],
                  redistribution=DEFAULT_MAP_CONFIG["pop_redistribution"],
                  noise_type = 'perlin',
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
        self.noise_type = noise_type

        if noise_type == 'simplex':
            self.fns = pyfastnoisesimd.Noise(seed=seed, numWorkers=n_workers)
            self.fns.noiseType = pyfastnoisesimd.NoiseType.Simplex
        else:
            gen = torch.Generator(device=device).manual_seed(seed)
            self.fns = FractalPerlin2D((1,width,width), [(1,)*2], [.5], generator=gen)

        self.dtype = dtype
        self.device = device

    def __call__(self):
        if self.noise_type == 'simplex':
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
        else:
            z = self.fns()[0]

        # Control diff between peaks and troughs
        z = (z - z.min()) / (z.max() - z.min())  # norm from 0 to 1
        z = torch.pow(z, self.redistribution)

        # Quantize to specific terrain levels
        z = torch.bucketize(z, torch.linspace(z.min(), z.max() + 0.0000001, self.levels + 1, dtype=self.dtype, device=self.device))
        z = (z - z.min()) / (z.max() - z.min())  # norm from 0 to 1

        # Resolve map to classes
        return torch.where(z >= self.urban_threshold, TerrainType.Urban, TerrainType.Rural)
