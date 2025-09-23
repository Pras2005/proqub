# ProQub: A Hyperspectral Cube Processor

**ProQub** is a focused Python library for processing hyperspectral data cubes. It provides a memory-efficient, streaming pipeline for common tasks like radiance-to-reflectance conversion and destriping, making it suitable for large datasets.

Originally designed for lunar IIRS data, its functions are named agnostically to be useful for a wide range of hyperspectral processing tasks.

## Key Features
- Memory-efficient processing for very large data cubes.
-  Radiance-to-reflectance conversion.
-  Two-pass destriping algorithm (median or mean).
- ⚙️ Flexible output formats (BSQ, BIL, BIP).
-  Granular control over each processing step.

## Installation

Install the package directly from PyPI:
```bash
pip install proqub

from proqub import run_pipeline

# 1. Define paths to your data files
hdr_path = 'path/to/your_radiance.hdr'
data_path = 'path/to/your_radiance.qub'
flux_data_path = 'path/to/solar_flux.txt'
geometric_param_path = 'path/to/your_geometry.spm'

# 2. Run the full processing pipeline
reflectance_path, destriped_path = run_pipeline(
    hdr_path=hdr_path,
    data_path=data_path,
    flux_data_path=flux_data_path,
    geometric_param_path=geometric_param_path,
    output_dir="./processed_data"
)

print(f"Processing complete!")
print(f"Final destriped cube saved to: {destriped_path}")

from proqub import CubeProcessor

# 1. Instantiate the processor
processor = CubeProcessor()

# 2. Open the source cube
radiance_image = processor.open_cube('path/to/radiance.hdr', 'path/to/radiance.qub')

# 3. Load ancillary data
flux = processor.load_flux_data('path/to/flux.txt')
angle = processor.parse_geometric_param('path/to/geometry.spm')

# 4. Perform only the radiance-to-reflectance step
processor.radiance_to_reflectance(
    radiance_img=radiance_image,
    output_path_base='processed/reflectance_only',
    flux_data=flux,
    incidence_angle_deg=angle
)

print("Reflectance conversion is complete.")
