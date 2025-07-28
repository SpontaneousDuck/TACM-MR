# TACM Dataset: Topographically-Augmented Channel Model Dataset

The TACM (Topographically-Augmented Channel Model) Dataset provides realistic channel models for automatic modulation recognition (AMR) research by combining modulated signals from the CSPB.ML.2018R2.nonoise dataset with realistic channel effects derived from terrain-based propagation models.

## 🎯 Overview

Traditional AMR datasets often use simplified channel models that don't reflect real-world propagation conditions. The TACM dataset addresses this limitation by:

- **Realistic Channel Effects**: Incorporates terrain-based 3GPP channel models with generated topography
- **Diverse Scenarios**: Supports multiple receiver configurations over generated terrain
- **Easy Integration**: Provides a PyTorch Lightning DataModule for seamless integration into ML workflows
- **Common Modulations**: Includes 8 modulation types with varying SNR conditions (-30 to +10 dB)

## 📦 Installation

### Option 1: Using Conda Environment (Recommended)
```bash
git clone https://github.com/mroftei/TACM-Dataset.git
cd TACM-Dataset
conda env create -f environment.yaml
conda activate tacm
pip install -e .
```

### Option 2: Install from Source
```bash
git clone https://github.com/mroftei/TACM-Dataset.git
cd TACM-Dataset
pip install -e .
```


## 🚀 Quick Start

### Basic Usage

```python
from tacm_dataset import TACMDataModule

# Initialize the data module
dm = TACMDataModule(
    dataset_root="~/data/tacm",  # Where to store/find data
    batch_size=32,
    download=True,    # Download CSPB dataset on first use
    gen_chan=True,    # Generate channel files on first use
    scenario='a'      # Single-receiver scenario
)

# Prepare and setup data (run once)
dm.prepare_data()  # Downloads and processes base dataset
dm.setup()         # Creates train/val/test splits

# Get data loaders
train_loader = dm.train_dataloader()
val_loader = dm.val_dataloader()
test_loader = dm.test_dataloader()

# Use in training loop
for batch in train_loader:
    signals = batch['x']           # Signal data [batch, channels, samples]
    labels = batch['y']            # Modulation labels [batch]
    snr = batch['snr_total']       # SNR values [batch]
    # ... your training code
```

### Advanced Configuration

```python
dm = TACMDataModule(
    dataset_root="~/data/tacm",
    batch_size=64,
    download=True,
    gen_chan=True,
    scenario='b',              # Multi-receiver with subsampling
    n_rx=4,                   # Number of receivers
    frame_size=512,           # Signal frame size
    snr_range=[-20, 5],       # SNR range in dB
    f_c=2.4e9,               # Carrier frequency (2.4 GHz)
    bw=20e3,                 # Bandwidth (20 kHz)
    n_topo=10000,            # Number of topographic scenarios
    seed=42                  # Random seed for reproducibility
)
```

## 📊 Dataset Details

### Modulation Types
The dataset includes 8 modulation schemes:
- **BPSK**: Binary Phase Shift Keying
- **QPSK**: Quadrature Phase Shift Keying  
- **8PSK**: 8-ary Phase Shift Keying
- **DQPSK**: Differential Quadrature Phase Shift Keying
- **MSK**: Minimum Shift Keying
- **16QAM**: 16-ary Quadrature Amplitude Modulation
- **64QAM**: 64-ary Quadrature Amplitude Modulation
- **256QAM**: 256-ary Quadrature Amplitude Modulation

### Scenarios
- **Scenario 'a'**: Standard single-receiver setup (main dataset)
- **Scenario 'b'**: Multi-receiver with signal subsampling
- **Scenario 'c'**: Multi-receiver flattened format
- **Scenario 'd'**: Multi-receiver with variable receiver count

### Data Splits
- **Training**: 60% of data
- **Validation**: 20% of data  
- **Testing**: 20% of data

## 🏗️ Dataset Structure

When using the dataset, each batch contains:

```python
batch = {
    'x': torch.Tensor,           # Signal data [batch, channels, samples]
    'y': torch.Tensor,           # Modulation labels [batch]
    'h_t': torch.Tensor,         # Channel impulse response
    'snr': torch.Tensor,         # Per-receiver SNR [batch, receivers]
    'snr_total': torch.Tensor,   # Total SNR [batch]
    'T_s': torch.Tensor,         # Symbol period [batch]
    'beta': torch.Tensor,        # Roll-off factor [batch]
    'tx_xy': torch.Tensor,       # Transmitter coordinates
    'rx_xy': torch.Tensor,       # Receiver coordinates
    'pow_rx': torch.Tensor,      # Received power [batch, receivers]
}
```

## 🔧 Configuration Options

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dataset_root` | - | Root directory for dataset files |
| `batch_size` | - | Mini-batch size |
| `download` | `True` | Download base CSPB dataset |
| `gen_chan` | `True` | Generate new channel files |
| `scenario` | `'a'` | Dataset scenario ('a', 'b', 'c', 'd') |
| `frame_size` | `1024` | Signal frame size (must divide 32768) |
| `n_rx` | `8` | Number of receivers |
| `snr_range` | `[-30, 10]` | SNR range in dB |
| `f_c` | `900e6` | Carrier frequency in Hz |
| `bw` | `30e3` | Signal bandwidth in Hz |
| `n_topo` | `28000` | Number of topographic scenarios |
| `seed` | `43` | Random seed |

---

**Note**: First-time setup requires downloading ~28GB of base signal data and generating channel files, which may take several hours depending on configuration.
