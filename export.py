
import rasterio

def export_raster(output_path, array, reference_profile):
    profile = reference_profile.copy()
    profile.update({
        'count': 1,
        'dtype': 'int32',
        'compress': 'lzw'
    })

    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(array.astype('int32'), 1)
