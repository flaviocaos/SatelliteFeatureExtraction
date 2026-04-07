# core/export.py
"""
Módulo de exportação: rasters classificados e figuras.
"""

import os
import numpy as np
import rasterio
from rasterio.transform import from_bounds


def export_raster(
    output_path: str,
    array: np.ndarray,
    reference_profile: dict,
) -> str:
    """
    Exporta array 2D como raster GeoTIFF com compressão LZW.
    Preserva a georreferenciação do raster de referência.

    Args:
        output_path: Caminho de saída (.tif)
        array: Array 2D (height, width)
        reference_profile: Profile rasterio da imagem de referência

    Returns:
        output_path confirmado
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    profile = reference_profile.copy()
    profile.update(
        dtype=rasterio.int32,
        count=1,
        compress="lzw",
        nodata=0,
    )

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(array.astype(np.int32), 1)

    return output_path


def export_figure(fig, output_path: str, dpi: int = 300) -> str:
    """
    Exporta figura matplotlib como PNG.

    Args:
        fig: Objeto Figure do matplotlib
        output_path: Caminho de saída (.png)
        dpi: Resolução da imagem

    Returns:
        output_path confirmado
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    return output_path
