from pathlib import Path

import numpy as np
import pyproj
from osgeo import gdal

# fmt: off
# Computing components of naive geotransform
x1, y1 = 4332750.0, 179936.0
x2, y2 = pyproj.Transformer.from_crs(
    26985, 3857, always_xy=True
).transform(x1, y1)
lng, lat = pyproj.Transformer.from_crs(
    26985, 4326, always_xy=True
).transform(x1, y1)
size = 1 / np.cos(np.radians(lat)) * 4.0

dataset: gdal.Dataset = gdal.Open(
    'smiley_merc.tif', gdal.GA_Update
)
geotransform = (x2, size, 0, y2, 0, -size)
dataset.SetGeoTransform(geotransform)
dataset.SetProjection(pyproj.CRS(3857).to_wkt())
dataset.FlushCache()
dataset = None
# fmt: on


# Simplified implementation of solving for best fit geotransform
def from_points(filename: Path, src_pts: np.ndarray, dst_pts: np.ndarray, wkt: str):
    matrix_a = []
    vec_b = []
    for src_pt, dst_pt in zip(src_pts, dst_pts):
        matrix_a.append([src_pt[0], src_pt[1], 1, 0, 0, 0])
        matrix_a.append([0, 0, 0, src_pt[0], src_pt[1], 1])
        vec_b.extend([dst_pt[0], dst_pt[1]])
    vec_x = np.linalg.lstsq(np.array(matrix_a), np.array(vec_b), rcond=None)[0]

    # GDAL ordering is different than affine transformation's
    geotransform = (vec_x[2], vec_x[0], vec_x[1], vec_x[5], vec_x[3], vec_x[4])

    dataset: gdal.Dataset = gdal.Open(str(filename), gdal.GA_Update)
    dataset.SetGeoTransform(geotransform)
    dataset.SetProjection(wkt)
    dataset.FlushCache()
    dataset = None
