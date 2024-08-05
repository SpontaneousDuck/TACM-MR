import sys
sys.path.insert(1, ".")
sys.path.insert(2, "../")
import h5py
import torch
from tqdm import tqdm
from tacm_dataset.scenario_gen import ScenarioGenerator
from tacm_dataset.sionna_torch import SionnaScenario
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--nrx", type=int, default=1)
parser.add_argument("--cspb", type=str, default="./CSPB.ML.2018R2_nonoise.hdf5")
parser.add_argument("--out", type=str, default="./")
args = parser.parse_args()

frame_size = 1024
n_rx = args.nrx
f_c = 900e6
bw = 30e3
seed=42
dev = torch.device('cuda:0')
dataset_path = args.cspb

print("Loading Data")
with h5py.File(dataset_path, "r") as f:
    x = torch.from_numpy(f['x'][()])
    y = torch.from_numpy(f['y'][()]).to(torch.long)
    t0 = torch.from_numpy(f['T0'][()]).to(torch.long)
    beta = torch.from_numpy(f['beta'][()])
    T_s = torch.from_numpy(f['T_s'][()])

print("Preprocessing Data")
# Make symbol index array
sym_idx = torch.linspace(0, 1.0, x.shape[-1]).repeat(x.shape[0], 1)
sym_idx *= (32768/T_s[:,None])
sym_idx = sym_idx.floor_().short()

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
sym_idx = sym_idx.reshape((-1, frame_size))
sym_idx -= sym_idx[:,:1]

# Scale signal to 0dB power
transmit_power = 0 # 0dB
signal_power = torch.mean(torch.abs(x) ** 2, -1)[...,None]
target_snr_linear = 10 ** (transmit_power / 10)
# occupied_bw = 1 / np.repeat(t0[i:i+step], x1.shape[0]/step)[:,None,None,None]
# signal_scale_linear = np.sqrt((target_snr_linear * occupied_bw) / signal_power)
signal_scale_linear = torch.sqrt((target_snr_linear) / signal_power)
x = x * signal_scale_linear

print("Generating Channels")
map_size = 1024
map_res = 20
scenario_gen = ScenarioGenerator(n_receivers=n_rx, 
                                 batch_size=128, 
                                 resolution=map_size, 
                                 min_receiver_dist=2,
                                 seed=42,
                                 dtype=x.dtype.to_real(), 
                                 device=dev)

batch_size = 128
sionna = SionnaScenario(n_bs=n_rx, 
                        batch_size=batch_size, 
                        n_time_samples=1024, 
                        f_c=900e6, 
                        bw=30e3, 
                        seed=seed, 
                        dtype=x.dtype, 
                        device=dev)

x = x.repeat((1,n_rx,1))
snr = torch.empty(len(x), n_rx, device=torch.device('cpu'))
snr_inband = torch.empty(len(x), n_rx, device=torch.device('cpu'))

for i in tqdm(range(0, len(x), batch_size), miniters=100, mininterval=10):
    scenario_gen.RegenerateFullScenario()
    sionna.update_topology(scenario_gen.transmitters, scenario_gen.receivers, scenario_gen.map, map_resolution=2)
    z, snr1 = sionna(x[i:i+batch_size,:1,None].to(dev))

    z = z.squeeze(2,3)[...,sionna.l_min*-1:sionna.l_max*-1]

    # Per-frame normalize to -1.0:1.0
    new_min, new_max = -1.0, 1.0
    z_max = torch.amax(torch.abs(z), axis=(1,2), keepdims=True) # farthest value from 0 in each channel
    scale = ((new_max - new_min) / (z_max*2))
    z *= scale

    bw = 10*torch.log10((1/T_s[i:i+batch_size])*(1+beta[i:i+batch_size]))
    snr1_inband = snr1.cpu() - bw[:,None,None]

    x[i:i+batch_size] = z.cpu()
    snr[i:i+batch_size] = snr1.flatten(1).cpu()
    snr_inband[i:i+batch_size] = snr1_inband.flatten(1).cpu()

h5py_path = f"{args.out}/TACM_2024_1.h5py"

with h5py.File(h5py_path, "w") as f:
    f['x'] = x.numpy(force=True)
    f['y'] = y
    f['snr'] = snr.numpy(force=True)
    f['snr_inband'] = snr_inband.numpy(force=True)
    # f['T0'] = t0
    f['T_s'] = T_s
    f['S_idx'] = sym_idx

        