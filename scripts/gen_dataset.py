import sys
sys.path.insert(1, ".")
sys.path.insert(2, "../")
import h5py
import torch
from tqdm import tqdm
from tacm_dataset.scenario_gen import ScenarioGenerator
from argparse import ArgumentParser
import numpy as np

parser = ArgumentParser()
parser.add_argument("--nrx", type=int, default=8)
parser.add_argument("--scen", type=str, default="a")
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
                                 map_size=map_size, 
                                 map_resolution=map_res, 
                                 min_receiver_dist=2, 
                                 max_iter=100, 
                                 frame_size=frame_size,
                                 f_c=900e6,
                                 bw=30e3,
                                 noise_type='perlin',
                                 seed=42,
                                 target_total_p=True if args.scen == "e" else False, 
                                 dtype=x.dtype.to_real(), 
                                 device=dev)

batch_size = 128
# x = x.repeat((1,n_rx,1))
snr = torch.empty(len(x), n_rx, device=torch.device('cpu'))
snr_inband = torch.empty(len(x), n_rx, device=torch.device('cpu'))
pow_rx = torch.empty(len(x), n_rx, device=torch.device('cpu'))
total_snr = torch.empty(len(x), device=torch.device('cpu'))
total_snr_inband = torch.empty(len(x), device=torch.device('cpu'))
h_t = torch.empty(len(x)//batch_size, n_rx, 1, 1, 1, frame_size+13, 14, device=torch.device('cpu'), dtype=x.dtype)
snr_gen = torch.Generator().manual_seed(seed)
snr_min = -30
snr_max = 10

if args.scen == "e" :
    target_snr = torch.empty(1)
else: 
    target_snr = torch.empty(n_rx)
for i in tqdm(range(0, len(x)//batch_size), miniters=100, mininterval=10):

    target_snr = target_snr.uniform_(snr_min, snr_max, generator=snr_gen)
    if args.scen == "d":
        target_power = target_snr + 10*np.log10((10**(scenario_gen.chan_gen.get_noise_power()/10))*n_rx)
        target_power[:] = 10*torch.log10((10**(target_power[0]/10))/n_rx)
    elif args.scen == "e":
        target_power = target_snr + 10*np.log10((10**(scenario_gen.chan_gen.get_noise_power()/10))*n_rx)
    else:
        target_power = target_snr + scenario_gen.chan_gen.get_noise_power()
        
    h_t[i] = scenario_gen.RegenerateFullScenario(target_power)
    # z, rx_pow_db, snr1 = scenario_gen.chan_gen.sionna.apply_channels(x[i:i+batch_size,:1,None].to(scenario_gen.chan_gen.sionna.device), h_t.to(scenario_gen.chan_gen.sionna.device), None)

    # z = z.squeeze(2,3)[...,scenario_gen.chan_gen.sionna.l_min*-1:scenario_gen.chan_gen.sionna.l_max*-1]

    # # Per-frame normalize to -1.0:1.0
    # new_min, new_max = -1.0, 1.0
    # z_max = torch.amax(torch.abs(z), axis=(1,2), keepdims=True) # farthest value from 0 in each channel
    # scale = ((new_max - new_min) / (z_max*2))
    # z *= scale

    # p_total_dbm = 10*torch.log10(torch.sum(10**(rx_pow_db.flatten(1)/10), 1))
    # p_noise_dbm = rx_pow_db - snr1
    # snr_total = p_total_dbm - (10*torch.log10((10**(p_noise_dbm/10)).sum((1,2))))
    # bw = 10*torch.log10((1/T_s[i:i+batch_size])*(1+beta[i:i+batch_size]))
    # snr1_inband = snr1.cpu() - bw[:,None,None]
    # snr_total_inband = snr_total.cpu() - bw # inband snr

    # x[i:i+batch_size] = z.cpu()
    # pow_rx[i:i+batch_size] = rx_pow_db.flatten(1).cpu()
    # snr[i:i+batch_size] = snr1.flatten(1).cpu()
    # snr_inband[i:i+batch_size] = snr1_inband.flatten(1).cpu()
    # total_snr[i:i+batch_size] = snr_total.cpu()
    # total_snr_inband[i:i+batch_size] = snr_total_inband 


h5py_path = f"{args.out}/TACM_2024_2_{args.nrx}rx_{args.scen}.h5py"

with h5py.File(h5py_path, "w") as f:
    f['x'] = x.numpy(force=True)
    f['y'] = y
    f['h_t'] = h_t.numpy(force=True)
    # f['p_rx'] = pow_rx.numpy(force=True)
    # f['snr'] = snr.numpy(force=True)
    # f['snr_inband'] = snr_inband.numpy(force=True)
    # f['snr_total'] = total_snr.numpy(force=True)
    # f['snr_total_inband'] = total_snr_inband.numpy(force=True)
    f['T0'] = t0
    f['beta'] = beta
    f['T_s'] = T_s
    f['S_idx'] = sym_idx

if args.scen == "c" or args.scen == "a" or args.scen == "b":
    for nrx in [8, 4, 2]:
        # Scenario B
        h5py_path = f"/work/esl/mroftei/datasets/TACM2024.2.1/TACM_2024_2_{nrx}rx_b.h5py"
        with h5py.File(h5py_path, "w") as f:
            f['x'] = x[::nrx].numpy(force=True)
            f['y'] = y[::nrx]
            f['h_t'] = h_t[::nrx, :nrx].numpy(force=True)
            f['T0'] = t0[::nrx]
            f['beta'] = beta[::nrx]
            f['T_s'] = T_s[::nrx]
            f['S_idx'] = sym_idx[::nrx]
            # f['p_rx'] = pow_rx[::nrx].numpy(force=True)
            # f['snr'] = snr[::nrx].numpy(force=True)
            # f['snr_inband'] = snr_inband[::nrx].numpy(force=True)
            # f['snr_total'] = total_snr[::nrx].numpy(force=True)
            # f['snr_total_inband'] = total_snr_inband[::nrx].numpy(force=True)

        # Scenario C
        h5py_path = f"/work/esl/mroftei/datasets/TACM2024.2.1/TACM_2024_2_{nrx}rx_c.h5py"
        with h5py.File(h5py_path, "w") as f:
            f['x'] = x.numpy(force=True)
            f['y'] = y
            f['h_t'] = h_t[:, :nrx].flatten(0,1).unsqueeze(1).numpy(force=True)
            f['T0'] = t0
            f['beta'] = beta
            f['T_s'] = T_s
            f['S_idx'] = sym_idx
            # f['x'] = x.repeat_interleave(nrx, dim=0).reshape(-1, 1, frame_size).numpy(force=True)
            # f['y'] = y.repeat_interleave(nrx, dim=0)
            # f['h_t'] = h_t[:, :nrx].flatten(0,1).unsqueeze(1).numpy(force=True)
            # f['T0'] = t0.repeat_interleave(nrx, dim=0)
            # f['beta'] = beta.repeat_interleave(nrx, dim=0)
            # f['T_s'] = T_s.repeat_interleave(nrx, dim=0)
            # f['S_idx'] = sym_idx.repeat_interleave(nrx, dim=0)
            # f['p_rx'] = pow_rx.reshape(-1, 1).numpy(force=True)
            # f['snr'] = snr.reshape(-1, 1).numpy(force=True)
            # f['snr_inband'] = snr_inband.reshape(-1, 1).numpy(force=True)
            # f['snr_total'] = snr.reshape(-1, 1).numpy(force=True)
            # bw = (total_snr - total_snr_inband)
            # f['snr_total_inband'] = (snr - bw[:,None]).reshape(-1, 1).numpy(force=True)
