"""
Setup script for the TACM Dataset package.

TACM (Topographically-Augmented Channel Model) Dataset provides realistic channel models
for automatic modulation recognition research by combining the CSPB.ML.2018R2 dataset
with terrain-based propagation effects.
"""

from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="tacm_dataset",
    version="1.0.0",
    description="Topographically-Augmented Channel Model Dataset for Automatic Modulation Recognition",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mroftei/TACM-Dataset",
    author="Ken Witham",
    author_email="k.witham@kri.neu.edu",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3 :: Only",
    ],
    keywords="machine-learning, signal-processing, channel-modeling, automatic-modulation-recognition, pytorch, deep-learning",
    packages=find_packages(),
    python_requires=">=3.8, <4",
    install_requires=[
        "torch>=2.0.0",
        "pytorch-lightning>=2.0.0",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "pandas>=1.3.0",
        "h5py>=3.6.0",
        "matplotlib>=3.5.0",
        "matplotlib-label-lines>=0.3.0",
        "tqdm>=4.60.0",
    ],
    extras_require={
        "dev": ["pytest>=6.0", "black", "flake8"],
        "fast": ["pyfastnoisesimd @ git+https://github.com/robbmcleod/pyfastnoisesimd.git@v0.4.3"],
    },
    project_urls={
        "Bug Reports": "https://github.com/mroftei/TACM-Dataset/issues",
        "Source": "https://github.com/mroftei/TACM-Dataset",
        "Documentation": "https://github.com/mroftei/TACM-Dataset/blob/main/README.md",
    },
)