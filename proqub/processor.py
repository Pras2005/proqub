"""
Hyperspectral Cube Processing Library (v2.1.0 - Simplified API)

A focused library for processing hyperspectral imagery. This version provides
a core CubeProcessor class for granular, step-by-step workflow control.
"""

import numpy as np
import pandas as pd
import spectral as spy
import os
import gc
from typing import Tuple, Optional

__version__ = "1.0.0"
__author__ = "Prasad, Aryan, Tanishka"


class CubeProcessor:
    """
    Main class for processing hyperspectral data cubes via a memory-efficient pipeline.
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.source_metadata = {}

    def _print(self, message: str):
        if self.verbose:
            print(message)

    def open_cube(self, hdr_path: str, data_path: str) -> spy.io.spyfile.SpyFile:
        """
        Opens an ENVI-format cube as a memory-mapped object.
        """
        if not os.path.exists(hdr_path) or not os.path.exists(data_path):
            raise FileNotFoundError("Header or data file not found")
        
        img = spy.envi.open(hdr_path, data_path)
        
        self.source_metadata = {
            'samples': img.shape[1],
            'lines': img.shape[0],
            'bands': img.shape[2],
            'byte order': img.byte_order,
            'interleave': img.interleave
        }
        
        self._print(f"Cube opened (not loaded). Shape: {img.shape}")
        return img

    def parse_geometric_param(self, file_path: str, fallback_value: float = 0.0) -> float:
        """
        Parses a text file to extract a single mean geometric parameter (e.g., incidence angle).
        """
        if not os.path.exists(file_path):
            self._print(f"Geometric param file not found. Using fallback: {fallback_value}")
            return fallback_value
        
        values = []
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    parts = line.split()
                    if len(parts) > 0:
                        try:
                            values.append(float(parts[-1]))
                        except ValueError:
                            continue
            
            if values:
                mean_val = np.mean(values)
                self._print(f"Parsed geometric parameter: {mean_val:.2f}")
                return mean_val
            else:
                self._print(f"No valid values found. Using fallback: {fallback_value}")
                return fallback_value
                
        except Exception as e:
            self._print(f"Error parsing file: {e}. Using fallback: {fallback_value}")
            return fallback_value

    def load_flux_data(self, file_path: str) -> np.ndarray:
        """
        Loads a two-column text file and returns the second column (e.g., solar flux).
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Flux data file not found: {file_path}")
        
        flux_data = np.loadtxt(file_path)
        flux_vector = flux_data[:, 1]
        
        self._print(f"Flux data loaded. Shape: {flux_vector.shape}")
        return flux_vector

    def radiance_to_reflectance(
        self,
        radiance_img: spy.io.spyfile.SpyFile,
        output_path_base: str,
        flux_data: np.ndarray,
        incidence_angle_deg: float,
        distance_au: float = 1.0,
        band_range: Tuple[int, int] = (5, 255),
        chunk_size: int = 256,
        interleave_format: str = 'bsq'
    ):
        """
        Converts a radiance cube to a reflectance cube.
        """
        self._print("Streaming radiance-to-reflectance conversion...")
        
        valid_formats = ['bsq', 'bil', 'bip']
        if interleave_format.lower() not in valid_formats:
            raise ValueError(f"Invalid interleave_format. Choose from {valid_formats}")

        cos_i = np.cos(np.deg2rad(incidence_angle_deg))
        eps = 1e-12
        
        start_band, end_band = band_range
        flux_data_cleaned = flux_data[start_band:end_band]
        
        lines, samples, _ = radiance_img.shape
        num_output_bands = end_band - start_band
        
        output_metadata = {
            'description': 'Reflectance Cube',
            'samples': str(samples),
            'lines': str(lines),
            'bands': str(num_output_bands),
            'data type': '4',
            'interleave': interleave_format,
            'file type': 'ENVI Standard',
            'byte order': self.source_metadata.get('byte order', 0)
        }
        
        output_hdr_path = output_path_base + '.hdr'
        os.makedirs(os.path.dirname(output_hdr_path), exist_ok=True)
        refl_file = spy.envi.create_image(
            output_hdr_path, output_metadata, ext='.qub', force=True
        )
        refl_mm = refl_file.open_memmap(writable=True)
        
        denominator = flux_data_cleaned[None, None, :] * cos_i * (distance_au**2) + eps
        
        for i in range(0, lines, chunk_size):
            chunk_end = min(i + chunk_size, lines)
            radiance_chunk = radiance_img[i:chunk_end, :, start_band:end_band]
            reflectance_chunk = (np.pi * radiance_chunk) / denominator
            refl_mm[i:chunk_end, :, :] = reflectance_chunk
        
        del refl_mm, refl_file
        gc.collect()
        self._print(f"Reflectance conversion complete. Saved to: {output_hdr_path}")

    def destripe_cube(
        self,
        input_img: spy.io.spyfile.SpyFile,
        output_path_base: str,
        method: str = 'median',
        chunk_size: int = 256,
        interleave_format: str = 'bsq'
    ):
        """
        Removes vertical striping from a cube using a two-pass algorithm.
        """
        self._print(f"Destriping cube using two-pass '{method}' method...")
        
        valid_formats = ['bsq', 'bil', 'bip']
        if interleave_format.lower() not in valid_formats:
            raise ValueError(f"Invalid interleave_format. Choose from {valid_formats}")

        lines, samples, bands = input_img.shape
        col_stats = np.zeros((bands, samples))

        for i in range(bands):
            band_view = input_img.read_band(i)
            if method == 'median':
                col_stats[i, :] = np.median(band_view, axis=0)
            elif method == 'mean':
                col_stats[i, :] = np.mean(band_view, axis=0)
            else:
                raise ValueError("Method must be 'median' or 'mean'")
        
        output_metadata = {
            'description': 'Destriped Cube',
            'samples': str(samples),
            'lines': str(lines),
            'bands': str(bands),
            'data type': '4',
            'interleave': interleave_format,
            'file type': 'ENVI Standard',
            'byte order': input_img.byte_order
        }
        
        output_hdr_path = output_path_base + '.hdr'
        os.makedirs(os.path.dirname(output_hdr_path), exist_ok=True)
        destriped_file = spy.envi.create_image(
            output_hdr_path, output_metadata, ext='.qub', force=True
        )
        destriped_mm = destriped_file.open_memmap(writable=True)
        
        for i in range(0, lines, chunk_size):
            chunk_end = min(i + chunk_size, lines)
            chunk = input_img[i:chunk_end, :, :]
            corrected_chunk = chunk - col_stats[None, :, :]
            destriped_mm[i:chunk_end, :, :] = corrected_chunk
            
        del destriped_mm, destriped_file
        gc.collect()
        self._print(f"Destriping complete. Saved to: {output_hdr_path}")
