# core/preprocessing.py
"""
Módulo de pré-processamento de imagens Sentinel-2.
Responsável por carregamento, normalização, NDVI e labels.
"""

import numpy as np
import rasterio
from rasterio.errors import RasterioIOError


def load_image(path: str) -> tuple[np.ndarray, dict]:
    """
    Carrega imagem raster multibanda.

    Args:
        path: Caminho para o arquivo .tif

    Returns:
        image: Array (bands, height, width) float32
        profile: Metadados rasterio do arquivo
    """
    try:
        with rasterio.open(path) as src:
            image = src.read().astype(np.float32)
            profile = src.profile.copy()
    except RasterioIOError as e:
        raise IOError(f"Não foi possível abrir o arquivo raster: {e}")
    return image, profile


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalização min-max por banda (0–1).
    Trata NaN e evita divisão por zero.

    Args:
        image: Array (bands, height, width)

    Returns:
        Array normalizado float32
    """
    image = image.astype(np.float32).copy()
    for i in range(image.shape[0]):
        band = image[i]
        min_val = np.nanmin(band)
        max_val = np.nanmax(band)
        rng = max_val - min_val
        if rng > 0:
            image[i] = (band - min_val) / rng
        else:
            image[i] = np.zeros_like(band)
    return image


def calculate_ndvi(nir_band: np.ndarray, red_band: np.ndarray) -> np.ndarray:
    """
    Calcula o Índice de Vegetação por Diferença Normalizada (NDVI).
    Clipa o resultado no intervalo [-1, 1].

    Args:
        nir_band: Banda NIR (height, width)
        red_band: Banda Red (height, width)

    Returns:
        ndvi: Array (height, width) com valores em [-1, 1]
    """
    nir = nir_band.astype(np.float32)
    red = red_band.astype(np.float32)
    ndvi = (nir - red) / (nir + red + 1e-6)
    return np.clip(ndvi, -1.0, 1.0)


def load_labels(label_path: str) -> tuple[np.ndarray, dict]:
    """
    Carrega raster de labels de classificação (banda 1).

    Args:
        label_path: Caminho para o arquivo .tif de labels

    Returns:
        labels: Array 2D (height, width) int
        profile: Metadados rasterio
    """
    try:
        with rasterio.open(label_path) as src:
            labels = src.read(1).astype(np.int32)
            profile = src.profile.copy()
    except RasterioIOError as e:
        raise IOError(f"Não foi possível abrir o arquivo de labels: {e}")
    return labels, profile
