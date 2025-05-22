
import rasterio
import numpy as np

def load_image(path):
    with rasterio.open(path) as src:
        image = src.read()
        profile = src.profile
    return image, profile

def normalize_image(image):
    image = image.astype(np.float32)
    for i in range(image.shape[0]):
        band = image[i]
        min_val, max_val = np.nanmin(band), np.nanmax(band)
        if max_val - min_val != 0:
            image[i] = (band - min_val) / (max_val - min_val)
    return image

def calculate_ndvi(nir_band, red_band):
    ndvi = (nir_band - red_band) / (nir_band + red_band + 1e-6)
    return ndvi

def load_labels(label_path):
    with rasterio.open(label_path) as src:
        labels = src.read(1)
        profile = src.profile
    return labels, profile
