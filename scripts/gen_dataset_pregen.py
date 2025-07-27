#!/usr/bin/env python3
"""
Script to generate and store all splits of TACM dataset for scenario 'a'.

This script uses TACMDataModule to generate the first epoch of train/validation/test
data and stores all data stacked by type into separate HDF5 datasets. The data is
fully processed through the TACM pipeline including channel effects, noise addition,
normalization, and SNR calculation via the on_after_batch_transfer method.

For example, all 'x' values across all batches are stacked into a single 'x' dataset,
all 'y' values into a 'y' dataset, and so forth.

The script generates three separate HDF5 files:
- *_train.h5: Training split (60% of data)
- *_val.h5: Validation split (20% of data) 
- *_test.h5: Test split (20% of data)

Requirements:
- Install the conda environment: conda env create -f environment.yaml
- Activate the environment: conda activate tacm
- Install the package: pip install -e .

Usage:
    # Basic usage with default parameters
    python gen_dataset_pregen.py
    
    # Download and generate data if needed
    python gen_dataset_pregen.py --download --gen_chan
    
    # Specify custom paths, output directory and batch size
    python gen_dataset_pregen.py --dataset_root ~/my_data --output_dir ~/results --batch_size 64

Output Structure:
    Each HDF5 file will contain datasets like:
    - 'x': All signal data stacked (shape: [total_samples, ...]) - fully processed with channel effects  
    - 'y': All labels stacked (shape: [total_samples])
    - 'snr': All SNR values stacked (shape: [total_samples, ...]) - calculated per-receiver
    - 'snr_total': All total SNR values stacked (shape: [total_samples])
    - 'pow_rx': All received power values stacked (shape: [total_samples, ...])
    - 'h_t': All channel responses stacked (shape: [total_samples, ...])
    - etc.
"""

import os
import sys
import argparse

# Add the parent directory to the path to import TACMDataModule
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    import h5py
    import torch
    import numpy as np
    from tqdm import tqdm
except ImportError as e:
    print(f"ERROR: Missing required dependency: {e}")
    print("Please install the required dependencies:")
    print("1. Create conda environment: conda env create -f environment.yaml")
    print("2. Activate environment: conda activate tacm")
    print("3. Install package: pip install -e .")
    sys.exit(1)

try:
    from tacm_dataset.TACMDataModule import TACMDataModule
except ImportError as e:
    print(f"ERROR: Cannot import TACMDataModule: {e}")
    print("Please ensure you have installed the package:")
    print("pip install -e .")
    sys.exit(1)


def process_dataloader(dataloader, split_name, dm):
    """
    Process a dataloader and return stacked data.
    
    Args:
        dataloader: PyTorch DataLoader to process
        split_name: Name of the split ('train', 'val', 'test') for progress display
        dm: TACMDataModule instance to call on_after_batch_transfer
        
    Returns:
        dict: Dictionary with stacked numpy arrays for each data key
        int: Number of batches processed
    """
    stacked_data = {}
    batch_count = 0

    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Processing {split_name} batches")):
        batch_count += 1

        for key, value in batch.items():
            batch[key] = value.to(dm.trainer.strategy.root_device)  # Ensure data is on the correct device
        
        # Apply channel effects and processing via on_after_batch_transfer
        processed_batch = dm.on_after_batch_transfer(batch, dataloader_idx=0)
        del processed_batch['h_t']

        # Convert tensors to numpy arrays and stack by key
        for key, value in processed_batch.items():
            if isinstance(value, torch.Tensor):
                numpy_value = value.detach().cpu().numpy()
                
                if key not in stacked_data:
                    stacked_data[key] = []
                stacked_data[key].append(numpy_value)
            else:
                # Handle scalar/non-tensor values
                if key not in stacked_data:
                    stacked_data[key] = []
                stacked_data[key].append(value)
    
    # Stack all arrays along the first dimension (batch dimension)
    final_data = {}
    for key, data_list in tqdm(stacked_data.items(), desc=f"Stacking {split_name} arrays"):
        if isinstance(data_list[0], np.ndarray):
            # Stack numpy arrays along batch dimension
            final_data[key] = np.concatenate(data_list, axis=0)
        else:
            # For scalar values, convert to numpy array
            final_data[key] = np.array(data_list)
    
    return final_data, batch_count


def save_split_to_hdf5(final_data, batch_count, output_path, split_name, dm, args):
    """
    Save processed split data to HDF5 file.
    
    Args:
        final_data: Dictionary with stacked numpy arrays
        batch_count: Number of batches processed
        output_path: Path to output HDF5 file
        split_name: Name of the split ('train', 'val', 'test')
        dm: TACMDataModule instance for metadata
        args: Command line arguments
    """
    print(f"Saving {split_name} split to: {output_path}")
    
    with h5py.File(output_path, 'w') as f:
        # Store metadata
        f.attrs['split'] = split_name
        f.attrs['scenario'] = 'a'
        f.attrs['batch_size'] = args.batch_size
        f.attrs['num_batches'] = batch_count
        f.attrs['total_samples'] = final_data['x'].shape[0] if 'x' in final_data else 0
        f.attrs['frame_size'] = dm.frame_size
        f.attrs['n_rx'] = dm.n_rx
        f.attrs['seed'] = dm.seed
        f.attrs['f_c'] = dm.f_c
        f.attrs['bw'] = dm.bw
        f.attrs['snr_range'] = dm.snr_range
        f.attrs['generation_date'] = str(np.datetime64('now'))
        
        # Create datasets for each data type
        for key, data in tqdm(final_data.items(), desc=f"Writing {split_name} datasets to HDF5"):
            if isinstance(data, np.ndarray):
                # Use compression for large arrays
                if data.size > 1000:
                    f.create_dataset(key, data=data, compression='gzip', compression_opts=6)
                else:
                    f.create_dataset(key, data=data)
            else:
                # Store scalar values as attributes (though this shouldn't happen with stacked data)
                f.attrs[f'{key}_value'] = data


def main():
    parser = argparse.ArgumentParser(
        description='Generate all splits of TACM dataset for scenario a',
        epilog="""
Examples:
  %(prog)s --download --gen_chan                    # Download and generate all data
  %(prog)s --dataset_root ~/my_data --batch_size 64 # Custom path and batch size
  %(prog)s --output_prefix my_data                  # Custom output file prefix
  %(prog)s --output_dir /path/to/output             # Custom output directory
  %(prog)s --output_dir ~/results --output_prefix experiment_1  # Custom directory and prefix
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--dataset_root', type=str, default='./data', 
                        help='Root directory for dataset files (default: ./)')
    parser.add_argument('--output_dir', type=str, default='./data',
                        help='Directory to save output HDF5 files (default: current directory)')
    parser.add_argument('--output_prefix', type=str, default='tacm_scenario_a',
                        help='Output HDF5 file prefix (default: tacm_scenario_a). Files will be named <prefix>_train.h5, <prefix>_val.h5, <prefix>_test.h5')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size for data loading (default: 1024)')
    parser.add_argument('--download', action='store_true',
                        help='Download base CSPB dataset if missing')
    parser.add_argument('--gen_chan', action='store_true', 
                        help='Generate new channel files')
    
    args = parser.parse_args()

    
    print("=" * 60)
    print("TACM Dataset All Splits Generation")
    print("=" * 60)
    print("TACM Dataset All Splits Generation")
    print("=" * 60)
    print(f"Dataset root: {os.path.expanduser(args.dataset_root)}")
    print(f"Output directory: {os.path.expanduser(args.output_dir)}")
    print(f"Output files: {args.output_prefix}_train.h5, {args.output_prefix}_val.h5, {args.output_prefix}_test.h5")
    print(f"Batch size: {args.batch_size}")
    print("Scenario: a (Standard single-receiver setup)")
    print("Processing: Full TACM pipeline with channel effects, noise, and normalization")
    print("Data will be stacked across batches into separate HDF5 datasets")
    print("Splits: train (60%), validation (20%), test (20%)")
    print("=" * 60)

    # Create output directory if it doesn't exist
    output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Created output directory: {output_dir}")

    print("Initializing TACMDataModule with scenario 'a' and default parameters...")
    
    # Initialize TACMDataModule with scenario 'a' and default parameters
    dm = TACMDataModule(
        dataset_root=args.dataset_root,
        batch_size=args.batch_size,
        download=args.download,
        gen_chan=args.gen_chan,
        scenario='a',  # Scenario 'a': Standard single-receiver setup
        # All other parameters use defaults as specified in TACMDataModule
        n_topo=28000,
        in_memory=True,
        frame_size=1024,
        f_c=900e6,
        bw=30e3,
        reuse_factor=128,
        gen_batch_size=128,
        snr_inband=True,
        snr_range=[-30, 10],
        seed=43,
        n_rx=8,
        n_workers=8
    )

    class Empty:
        pass
    dm.trainer = Empty()
    dm.trainer.strategy = Empty()
    dm.trainer.strategy.root_device = torch.device('cuda')

    print("Preparing data (downloading/processing if needed)...")
    dm.prepare_data()
    
    print("Setting up dataset splits...")
    dm.setup()
    
    # Get all dataloaders
    print("Getting all dataloaders...")
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    test_loader = dm.test_dataloader()
    
    print(f"Total batches - Train: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)}")
    
    # Process each split
    splits = [
        ('train', train_loader),
        ('val', val_loader),
        ('test', test_loader)
    ]
    
    all_stats = {}
    
    for split_name, dataloader in splits:
        # Process the split - pass dm to enable on_after_batch_transfer processing
        final_data, batch_count = process_dataloader(dataloader, split_name, dm)
        
        # Save to HDF5 file
        output_path = os.path.join(os.path.expanduser(args.output_dir), f"{args.output_prefix}_{split_name}.h5")
        save_split_to_hdf5(final_data, batch_count, output_path, split_name, dm, args)
        
        # Store statistics
        all_stats[split_name] = {
            'final_data': final_data,
            'batch_count': batch_count,
            'output_path': output_path
        }
        
        print(f"Successfully saved {split_name} split with {batch_count} batches to {output_path}")
    
    # Print comprehensive statistics
    print("\n" + "="*60)
    print("GENERATION COMPLETE - SUMMARY")
    print("="*60)
    
    for split_name, stats in all_stats.items():
        final_data = stats['final_data']
        batch_count = stats['batch_count']
        output_path = stats['output_path']
        
        print(f"\n{split_name.upper()} SPLIT:")
        total_samples = final_data['x'].shape[0] if 'x' in final_data else 0
        print(f"  Total samples: {total_samples}")
        print(f"  Number of batches: {batch_count}")
        
        if 'x' in final_data:
            print(f"  Signal data shape: {final_data['x'].shape}")
            print(f"  Signal data type: {final_data['x'].dtype}")
        if 'y' in final_data:
            print(f"  Label data shape: {final_data['y'].shape}")
            print(f"  Label data type: {final_data['y'].dtype}")
        if 'snr' in final_data:
            print(f"  SNR data shape: {final_data['snr'].shape}")
            print(f"  SNR data type: {final_data['snr'].dtype}")
        if 'h_t' in final_data:
            print(f"  Channel data shape: {final_data['h_t'].shape}")
            print(f"  Channel data type: {final_data['h_t'].dtype}")
        
        print(f"  Available datasets: {list(final_data.keys())}")
        
        # Calculate file size
        file_size_bytes = os.path.getsize(output_path)
        file_size_mb = file_size_bytes / (1024 * 1024)
        print(f"  File size: {file_size_mb:.2f} MB")
        print(f"  Output file: {output_path}")
    
    # Print total statistics
    total_samples = sum(stats['final_data']['x'].shape[0] if 'x' in stats['final_data'] else 0 
                        for stats in all_stats.values())
    total_file_size = sum(os.path.getsize(stats['output_path']) for stats in all_stats.values())
    total_file_size_mb = total_file_size / (1024 * 1024)
    
    print("\nOVERALL TOTALS:")
    print(f"  Total samples across all splits: {total_samples}")
    print(f"  Total file size: {total_file_size_mb:.2f} MB")
    print(f"  Original batch size: {args.batch_size}")
    
    print("\nAll splits generated successfully!")
        


if __name__ == '__main__':
    main()
