# ProQub (seaborne)

A Python library designed for processing hyperspectral data cubes using a memory-efficient streaming pipeline. 

## Overview
`proqub` (packaged internally as `seaborne`) is a specialized tool for converting raw radiance data from hyperspectral sensors into reflectance values. It is optimized to handle massive ENVI-format data cubes without overloading system RAM by utilizing chunked streaming operations. Originally developed for astronomical/remote sensing tasks (like lunar mapping), it integrates tightly with the `spectral` (Spectral Python) library.

As a secondary utility, the package includes an `MLExamples` class—a cheat-sheet repository of common Machine Learning and AI algorithms (A*, N-Queens, SVM, Logistic Regression, etc.) bundled together.

## Domain Model & Features
The primary processing logic lives within `CubeProcessor` in `seaborne/processor.py`:
- **Hyperspectral I/O**: Loads `.hdr` and binary ENVI files dynamically using `spy.envi.open`.
- **Geometric Parameter Parsing**: Extracts incidence angles or other sensor metadata from auxiliary text files to calculate exact solar incidence during conversion.
- **Radiance to Reflectance Pipeline**:
  - Operates via `radiance_to_reflectance()`.
  - Implements the standard physical formula correcting for solar irradiance (`flux_data`), planetary distance (`distance_au`), and the cosine of the solar incidence angle.
  - **Memory Efficiency**: Instead of loading a 10GB+ cube into memory, it processes spatial lines in small `chunk_size` batches.
  - Automatically writes the output to disk with updated `.hdr` metadata conforming to BSQ, BIL, or BIP interleave formats.

## Prerequisites
- Python 3.8+
- `numpy`
- `pandas`
- `spectral` (Spectral Python)
- `scikit-learn` (for the ML snippets module)

## Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone git@github.com:Pras2005/proqub.git
   cd proqub
   ```

2. **Install via pip**:
   ```bash
   pip install .
   ```
   *Alternatively, install from PyPI if published (e.g., `pip install proqub`).*

## Usage / Running Locally

**Hyperspectral Processing Example**:
```python
from seaborne import CubeProcessor
import numpy as np

# Initialize the processor
processor = CubeProcessor(verbose=True)

# Open the raw cube (does not load into RAM)
raw_cube = processor.open_cube('data.hdr', 'data.img')

# Load external Solar Flux data for calibration
flux_array = processor.load_flux_data('solar_flux.txt')

# Extract incidence angle from geometric sensor logs
inc_angle = processor.parse_geometric_param('geometry.txt', fallback_value=30.0)

# Stream and convert chunk-by-chunk to save memory
processor.radiance_to_reflectance(
    radiance_img=raw_cube,
    output_path_base='output_reflectance',
    flux_data=flux_array,
    incidence_angle_deg=inc_angle,
    chunk_size=128
)
```

## Project Structure
```text
proqub/
├── pyproject.toml         # Modern Python package configuration
├── setup.cfg              # Setup configurations
├── seaborne/              # Main package directory
│   ├── __init__.py        # Package exports and versioning
│   └── processor.py       # Core CubeProcessor logic & MLExamples classes
└── README.md              # Documentation
```
