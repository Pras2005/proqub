"""
ProQub: A Hyperspectral Cube Processing Library.

This package provides tools for common hyperspectral processing tasks
like radiance-to-reflectance conversion and destriping in a memory-efficient pipeline.
"""

from .processor import CubeProcessor, run_pipeline

__version__ = "1.0.0"
