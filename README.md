# Terrain Augmented Channel Model (TACM) Dataset
This repostory hosts the code for the TACM dataset for automatic modulation recognition. Code is provided for generating the dataset and a PyTroch Lightning DataModule is also provided for using the genreted dataset. INstructions for using are below.

## Usage
### Generate Dataset
The `scripts/gen_dataset.py` script will generate an HDF5 storing the generated dataset. The specially generated CSPB.ML.2018R2 dataset with no noise added is required as an input. `scripts/pac_CSPB2018RX.ipynb` aids in converting the format CSPB.ML.2018R2 no-noise is deliverd in to the packed HDF5 file required by this script.

### PyTorch Lightning DataModule
This folder can be installed with pip to be used in any PyTorch Lighting based training code. THe `DataModule` can be imported with `from tacm_dataset.TACMDataModule import TACMDataModule`. The output of `scripts/gen_dataset.py` is a HDF5 file which is the input to this DataModule.

## Publications
### TACM2024.1:
- K. L. Witham, N. M. Prabhu, A. Sultan, M. Necsoiu, C. Spooner, and G. Schirner, “Utilizing terrain-generation to derive realistic channel models for automatic modulation recognition,” in Synthetic Data for Artificial Intelligence and Machine Learning: Tools, Techniques, and Applications II, SPIE, Jun. 2024, pp. 450–461. doi: 10.1117/12.3013507.
