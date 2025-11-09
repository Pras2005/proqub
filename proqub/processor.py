"""
Hyperspectral Cube Processing Library (v2.2.0 - Simplified API + ML Demonstrations)

Provides a CubeProcessor for hyperspectral cube processing, along with MLExamples
for educational machine learning code snippets (PCA, KMeans, KNN, Linear Regression).
"""

import numpy as np
import pandas as pd
import spectral as spy
import os
import gc
from typing import Tuple, Optional

__version__ = "2.2.0"
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


# ==============================================================
# Machine Learning Demonstration Utilities
# ==============================================================

class MLExamples:
    """ Educational machine learning examples (prints working code). """

    def pca(self):
        print('''\
# PCA on Iris Dataset
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("iris.csv")
X = df.iloc[:, 1:5].values
y = df['Species'].values

X_scaled = StandardScaler().fit_transform(X)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
df_pca['Species'] = y

plt.figure(figsize=(8,6))
sns.scatterplot(x='PC1', y='PC2', hue='Species', data=df_pca, s=100)
plt.title('PCA of Iris Dataset')
plt.show()
''')

    def kmeans(self):
        print('''\
# K-Means Clustering with Elbow Method
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

df = pd.read_csv("iris.csv")
X = df.iloc[:, 1:5].values

wcss = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, random_state=0)
    km.fit(X)
    wcss.append(km.inertia_)

plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Clusters')
plt.ylabel('WCSS')
plt.show()
''')

    def knn(self):
        print('''\
# KNN Classification on Iris Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("iris.csv")
X = df.iloc[:, 1:5].values
y = LabelEncoder().fit_transform(df['Species'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
''')

    def linear_regression(self):
        print('''\
# Linear Regression (from scratch)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 5])

m = np.sum((x - x.mean()) * (y - y.mean())) / np.sum((x - x.mean())**2)
c = y.mean() - m * x.mean()

y_pred = m * x + c

print(f"Slope: {m:.3f}, Intercept: {c:.3f}")
print("R²:", r2_score(y, y_pred))
print("MSE:", mean_squared_error(y, y_pred))
print("MAE:", mean_absolute_error(y, y_pred))

plt.scatter(x, y, color='blue')
plt.plot(x, y_pred, color='red')
plt.title('Linear Regression (from scratch)')
plt.show()
''')

    def pandas_demo(self):
        print('''\
# Basic Pandas Operations
import pandas as pd

df = pd.read_csv("iris.csv")
print(df.head())
print("\\nColumns:", df.columns.tolist())
print("\\nBasic Statistics:\\n", df.describe())
print("\\nSpecies counts:\\n", df['Species'].value_counts())
''')


# Example Usage:
# >>> demo = MLExamples()
# >>> demo.pca()
# >>> demo.kmeans()
# >>> demo.linear_regression()

