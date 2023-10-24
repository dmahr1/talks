import dataclasses
import shutil
import subprocess as sp
from pathlib import Path
from typing import Optional

import numpy as np
import pyproj
from osgeo import gdal
from PIL import Image

np.set_printoptions(suppress=True, linewidth=150)
gdal.UseExceptions()


@dataclasses.dataclass
class Geotransform:
    """Container for a 2x3 GDAL Geotransform matrix: https://gdal.org/tutorials/geotransforms_tut.html"""

    ul_x: float
    pixel_width: float
    row_rotation: float
    ul_y: float
    col_rotation: float
    pixel_height: float

    def to_matrix(self) -> np.ndarray:
        """Get the geotransform as a 2x3 affine transform matrix in numpy form. This ordering differs from GDAL."""
        return np.array(
            [[self.pixel_width, self.row_rotation, self.ul_x], [self.col_rotation, self.pixel_height, self.ul_y]]
        )

    def to_gdal_tuple(self) -> tuple[float, ...]:
        """Get the geotransform as a 6-tuple in GDAL ordering"""
        return self.ul_x, self.pixel_width, self.row_rotation, self.ul_y, self.col_rotation, self.pixel_height

    @staticmethod
    def from_gdal_tuple(gdal_tuple: tuple[float, ...]) -> 'Geotransform':
        """Get a geotransform from a 6-tuple in GDAL ordering"""
        return Geotransform(
            ul_x=gdal_tuple[0],
            pixel_width=gdal_tuple[1],
            row_rotation=gdal_tuple[2],
            ul_y=gdal_tuple[3],
            col_rotation=gdal_tuple[4],
            pixel_height=gdal_tuple[5],
        )

    @staticmethod
    def from_path(path: Path) -> 'Geotransform':
        if not path.exists():
            raise FileNotFoundError(f'File {path} does not exist')
        dataset: gdal.Dataset = gdal.Open(str(path), gdal.GA_ReadOnly)
        geotransform = Geotransform.from_gdal_tuple(dataset.GetGeoTransform())
        dataset = None
        return geotransform

    @staticmethod
    def from_points(src_pts: np.ndarray, dst_pts: np.ndarray, allow_rotation: bool = True) -> 'Geotransform':
        """Estimate a geotransform (with 6 degrees of freedom) using a closed form least squares solver.

        Given a src point [x, y, 1]^T we want to find the 2x3 transform that produces a dst point [x', y']^T
        ┌           ┐   ┌     ┐   ┌     ┐
        │  a  b  c  │   │  x  │   │  x' │
        │  d  e  f  │ * │  y  │ = │  y' │
        └           ┘   │  1  │   └     ┘
                        └     ┘
        Given a collection of n src pts, we can solve for the transform parameters that best predict the n dst pts in a
        least squares sense. We have to rewrite this in the form AX=B where matrix A ∈ ℝ^(2n x 6) contains the src
        points, column vector X ∈ ℝ^(6 x 1) contains the transform elements, and column vector B ∈ ℝ^(2n x 1) contains
        the dst pts. This system of equations can then be solved for using numpy.
                     A                   X    =     B
        ┌                         ┐   ┌     ┐   ┌       ┐
        │ x_1 y_1  1   0   0   0  │   │  a  │   │  x_1' │
        │  0   0   0  x_1 y_1  1  │   │  b  │   │  y_1' │
        │ x_2 y_2  1   0   0   0  │   │  c  │   │  x_2' │
        │  0   0   0  x_2 y_2  1  │   │  d  │   │  y_2' │
        │                  . .    │   │  e  │   │  ...  │
        │ x_n y_n  1   0   0   0  │   │  f  │   │  x_n' │
        │  0   0   0  x_n y_n  1  │   └     ┘   │  y_n' │
        └                         ┘             └       ┘
        """
        if src_pts.shape != dst_pts.shape:
            raise ValueError('src_pts and dst_pts must be the same length')
        if src_pts.shape[0] < 4:
            raise ValueError(f'src_pts and dst_pts must have at least 4 points, {src_pts.shape[0]} found')

        matrix_a = []
        vector_b = []
        for src_pt, dst_pt in zip(src_pts, dst_pts):
            matrix_a.append([src_pt[0], src_pt[1], 1, 0, 0, 0] if allow_rotation else [src_pt[0], 1, 0, 0])
            matrix_a.append([0, 0, 0, src_pt[0], src_pt[1], 1] if allow_rotation else [0, 0, src_pt[1], 1])
            vector_b.extend([dst_pt[0], dst_pt[1]])
        solved = np.linalg.lstsq(np.array(matrix_a), np.array(vector_b), rcond=None)[0]
        if allow_rotation:  # output has a, b, c, d, e, f
            geotransform = Geotransform(
                pixel_width=solved[0],
                row_rotation=solved[1],
                ul_x=solved[2],
                col_rotation=solved[3],
                pixel_height=solved[4],
                ul_y=solved[5],
            )
        else:  # output has a, c, e, f
            geotransform = Geotransform(
                pixel_width=solved[0],
                row_rotation=0.0,
                ul_x=solved[1],
                col_rotation=0.0,
                pixel_height=solved[2],
                ul_y=solved[3],
            )
        return geotransform

    def transform(self, points_pixel: np.ndarray) -> np.ndarray:
        """Transform n x 2 matrix with pixel coordinates to n x 2 matrix of georeferenced coordinates"""
        point_matrix = np.hstack((points_pixel, np.ones((points_pixel.shape[0], 1)))).T  # homogeneous coordinates
        transformed = self.to_matrix() @ point_matrix
        return transformed.T  # output 1 row per point and 1 column per dimension

    def compute_error(self, src_pts: np.ndarray, dst_pts: np.ndarray) -> tuple[float, float]:
        """Compute the max and root mean squared error (RMSE) between the src_pts and the dst_pts"""
        points_geotransformed = self.transform(src_pts)
        errors = np.linalg.norm(dst_pts - points_geotransformed, axis=1)
        return (np.max(errors), np.sqrt(np.average(errors**2)))

    def report_error(self, src_pts: np.ndarray, dst_pts: np.ndarray):
        """Report the root mean squared error (RMSE) between the src_pts and the dst_pts"""
        error_max, error_rmse = self.compute_error(src_pts, dst_pts)
        avg_pixel_size = np.average(np.abs([self.pixel_height, self.pixel_width]))
        print(f'    Max: {error_max:.12f}, {error_max / avg_pixel_size:.3%} of pixel size')
        print(f'    RMSE: {error_rmse:.12f}, {error_rmse / avg_pixel_size:.3%} of pixel size')

    def apply_to_dataset(self, filename: Path, wkt: str):
        """Apply the current geotransform and WKT to the given dataset"""
        if not filename.exists():
            raise ValueError(f'Could not find {filename}')
        dataset: gdal.Dataset = gdal.Open(str(filename), gdal.GA_Update)
        # old_geotransform = dataset.GetGeoTransform()
        new_geotransform = self.to_gdal_tuple()
        print(f'  {filename}: {new_geotransform=}')
        dataset.SetGeoTransform(new_geotransform)
        dataset.SetProjection(wkt)
        dataset.FlushCache()
        dataset = None


@dataclasses.dataclass
class Raster:
    width: int
    height: int
    geotransform: Geotransform
    unique_values: int
    crs_name: str
    crs: pyproj.CRS = dataclasses.field(repr=False)
    units: str

    def from_path(path: Path) -> 'Raster':
        if not path.exists():
            raise FileNotFoundError(f'File {path} does not exist')
        dataset: gdal.Dataset = gdal.Open(str(path), gdal.GA_ReadOnly)
        band: gdal.Band = dataset.GetRasterBand(1)
        unique_values = np.unique(band.ReadAsArray())
        crs = pyproj.CRS(dataset.GetProjection())
        gdalinfo = Raster(
            width=dataset.RasterXSize,
            height=dataset.RasterYSize,
            geotransform=Geotransform.from_gdal_tuple(dataset.GetGeoTransform()),
            crs_name=crs.name,
            crs=crs,
            units=crs.axis_info[0].unit_name,
            unique_values=len(unique_values),
        )
        band = None
        dataset = None
        return gdalinfo

    def src_and_dst_points(self, approx_count: int, dst_crs: pyproj.CRS) -> tuple[np.ndarray, np.ndarray]:
        spacing = int(np.round(np.sqrt((self.width * self.height) / approx_count)))
        cols = np.arange(0, self.width, spacing)
        rows = np.arange(0, self.height, spacing)
        points_pixel = np.array(np.meshgrid(cols, rows)).T.reshape(-1, 2)
        points_src_crs = self.geotransform.transform(points_pixel)
        transformer = pyproj.Transformer.from_crs(self.crs, dst_crs, always_xy=True)
        points_dst_crs = np.array(transformer.transform(*points_src_crs.T)).T
        return points_pixel, points_dst_crs


def run_cmd(
    command: list[str], current_dir: Optional[Path] = None, timeout_seconds: float = 10.0, verbose: bool = True
) -> tuple[int, str]:
    try:
        command = [str(c) for c in command]
        if verbose:
            print(f'Running command: {" ".join(command)}')
        output = sp.check_output(
            command, timeout=timeout_seconds, cwd=str(current_dir) if current_dir else None, stderr=sp.STDOUT
        )
        output = output.decode().strip()
        if verbose and output:
            print(f'Output: {output}')
        return 0, output
    except sp.CalledProcessError as e:
        output = e.output.decode().strip()
        print(f'Error {e.returncode} running command, output: {output}')
        return e.returncode, output


class SampleData:
    """Dumb container to group some methods related to creating sample data for FOSS4GNA 2023 presentation."""

    @classmethod
    def upsample(cls, img: Image.Image, scale_factor: int = 8) -> Image.Image:
        """Helper method to upsample an image so that individual pixels are preserved when shown in a slideshow. By default,
        Google Slides applies some kind of bilinear or bicubic resampling; we effectively want nearest neighbor."""
        return img.resize(np.array(img.size) * scale_factor, Image.Resampling.NEAREST)

    @classmethod
    def paint_circle(
        cls,
        data: np.ndarray,
        center_x: float,
        center_y: float,
        outer_dist: float,
        inner_dist: float,
        start_azimuth: float = 0.0,
        end_azimuth: float = 360.0,
        color: int = 255,
    ) -> np.ndarray:
        rows, cols = data.shape
        for i in range(rows):
            for j in range(cols):
                dist = np.sqrt((i - center_y) ** 2 + (j - center_x) ** 2)
                if not (inner_dist <= dist <= outer_dist):
                    continue
                azimuth = (np.degrees(np.arctan2(i - center_y, j - center_x)) + 90) % 360
                if not (start_azimuth <= azimuth <= end_azimuth):
                    continue
                data[i, j] = color
        return data

    @classmethod
    def create_smiley_face_example_image(cls, dim: int) -> np.ndarray:
        smiley = np.zeros((dim, dim), dtype='uint8')
        smiley = cls.paint_circle(smiley, dim / 2, dim / 2, dim / 2, dim / 2 - 8)  # outer circle
        smiley = cls.paint_circle(smiley, 5 / 16 * dim, 5 / 16 * dim, 8, 0)  # left eye
        smiley = cls.paint_circle(smiley, 11 / 16 * dim, 5 / 16 * dim, 8, 0)  # right eye
        smiley = cls.paint_circle(smiley, dim / 2, 7 / 16 * dim, dim / 3, dim / 3 - 8, 135, 225)  # mouth
        smiley[np.arange(dim // 2, dim, 2), :] = 0  # horizontal lines in lower half
        smiley[:, np.arange(dim // 2, dim, 2)] = 0  # vertical lines in right half
        return smiley

    @classmethod
    def perform_lossless_operations(cls, image: Image.Image, prefix: str) -> Image.Image:
        cls.upsample(image).save(f'{prefix}01_upsampled.png')
        cls.upsample(image.transpose(Image.Transpose.ROTATE_90)).save(f'{prefix}02_rotate90.png')
        cls.upsample(image.transpose(Image.Transpose.ROTATE_180)).save(f'{prefix}03_rotate180.png')
        cls.upsample(image.transpose(Image.Transpose.ROTATE_270)).save(f'{prefix}04_rotate270.png')
        cls.upsample(image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)).save(f'{prefix}05_flipy.png')
        cls.upsample(image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)).save(f'{prefix}06_flipx.png')
        cls.upsample(image.transpose(Image.Transpose.TRANSPOSE)).save(f'{prefix}07_transpose.png')
        cls.upsample(image.resize(np.array([image.width, image.height]) * 2, Image.Resampling.NEAREST)).save(
            f'{prefix}08_2x_nearest.png'
        )
        cls.upsample(image.resize(np.array([image.width, image.height]) * 2, Image.Resampling.NEAREST)).save(
            f'{prefix}09_3x_nearest.png'
        )

    @classmethod
    def perform_lossy_operations(cls, image: Image.Image, dim: int, prefix: str) -> Image.Image:
        cls.upsample(image.resize((dim, dim), Image.Resampling.NEAREST)).save(f'{prefix}10_upsample_nearest.png')
        cls.upsample(image.resize((dim, dim), Image.Resampling.BILINEAR)).save(f'{prefix}11_upsample_bilinear.png')
        cls.upsample(image.resize((dim, dim), Image.Resampling.BICUBIC)).save(f'{prefix}12_upsample_bicubic.png')
        cls.upsample(image.resize((dim, dim), Image.Resampling.LANCZOS)).save(f'{prefix}13_upsample_lanczos.png')
        cls.upsample(image.rotate(45, Image.Resampling.NEAREST, True)).save(f'{prefix}14_rotate45_nearest.png')
        cls.upsample(image.rotate(45, Image.Resampling.BILINEAR, True)).save(f'{prefix}15_rotate45_bilinear.png')
        cls.upsample(image.rotate(45, Image.Resampling.BICUBIC, True)).save(f'{prefix}16_rotate45_bicubic.png')


if __name__ == '__main__':
    # Create sample image
    dim = 128
    smiley = Image.fromarray(SampleData.create_smiley_face_example_image(dim))
    step0_raw = Path('smiley00_raw.png')
    smiley.save(str(step0_raw))

    # Create images for slides demonstrating lossless vs. lossy operations
    SampleData.perform_lossless_operations(smiley, 'smiley')
    SampleData.perform_lossy_operations(smiley, dim * 4 // 3, 'smiley')

    # Convert PNG to TIFF
    step1_geotiff = Path('smiley20_geotiff.tif')
    run_cmd(['gdal_translate', '-of', 'GTiff', str(step0_raw), str(step1_geotiff)])

    # Apply georeferencing
    pixel_size = 4.0
    step2_georef = Path('smiley21_georef.tif')
    x, y = 432750, 179936  # near location of FOSS4GNA conference in Baltimore in EPSG:26985
    lng, lat = pyproj.Transformer.from_crs(26985, 4326, always_xy=True).transform(x, y)
    shutil.copy(step1_geotiff, step2_georef)
    run_cmd(
        [
            'gdal_edit.py',
            '-a_srs',
            'epsg:26985',
            '-a_ullr',
            x,
            y,
            x + dim * pixel_size,
            y - dim * pixel_size,
            str(step2_georef),
        ]
    )
    print(Raster.from_path(step2_georef))

    # Warp from NAD83 / Maryland to NAD83 / Maryland (ftUS); this has no distortion
    step3_warp_feet = Path('smiley22_warp_feet.tif')
    run_cmd(['gdalwarp', '-t_srs', 'epsg:2248', '-r', 'bilinear', str(step2_georef), str(step3_warp_feet)])
    print(Raster.from_path(step3_warp_feet))

    # Warp from NAD83 / Maryland to UTM and Web Mercator via various methods
    for code in [
        32618,  # UTM 18N
        3857,  # web mercator
        5070,  # CONUS albers
        # could add other coordinate systems here easily enough
    ]:
        crs = pyproj.CRS(f'epsg:{code}')
        print(f'\n==================== PROCESSING EPSG:{code}: {crs.name} ====================')
        raster = Raster.from_path(step2_georef)
        src_pts, dst_pts = raster.src_and_dst_points(1000, crs)
        print(raster)

        # Method 1: simply use gdalwarp; this has distortion
        step3_warped = Path(f'smiley23_warp_{code}.tif')
        run_cmd(['gdalwarp', '-t_srs', f'epsg:{code}', '-r', 'bilinear', str(step2_georef), str(step3_warped)])
        print(Raster.from_path(step3_warped))

        # Method 2: naively update the geotransform's origin and pixel size based purely on intuition
        step4_naive = Path(f'smiley24_naive_{code}.tif')
        shutil.copy(step2_georef, step4_naive)
        naive_geotransform = Geotransform.from_path(step4_naive)
        x2, y2 = pyproj.Transformer.from_crs(26985, code, always_xy=True).transform(x, y)
        naive_geotransform.ul_x = x2
        naive_geotransform.ul_y = y2
        x3, y3 = pyproj.Transformer.from_crs(26985, code, always_xy=True).transform(x + pixel_size, y + pixel_size)
        if code == 3857:
            mercator_scale_factor = 1 / np.cos(np.radians(lat))
            naive_geotransform.pixel_width = pixel_size * mercator_scale_factor
            naive_geotransform.pixel_height = pixel_size * mercator_scale_factor * -1
        else:
            naive_geotransform.pixel_width = abs(x3 - x2)
            naive_geotransform.pixel_height = abs(y3 - y2) * -1
        naive_geotransform.report_error(src_pts, dst_pts)
        naive_geotransform.apply_to_dataset(step4_naive, crs.to_wkt())

        # Method 3: numerically solve for geotransform with rotation and apply to raster. This does not render properly
        # in QGIS as discussed in https://gis.stackexchange.com/a/441243/4669 and in the QGIS issue tracker
        # https://github.com/qgis/QGIS/issues/23760#issuecomment-1303546604. It also does not work with some GDAL
        # operations. But it does render properly in ArcGIS.
        step5_solved = Path(f'smiley25_solved_{code}.tif')
        shutil.copy(step2_georef, step5_solved)
        solved_geotransform = Geotransform.from_points(src_pts, dst_pts)
        solved_geotransform.report_error(src_pts, dst_pts)
        solved_geotransform.apply_to_dataset(step5_solved, crs.to_wkt())
        SampleData.upsample(Image.open(step5_solved)).save(Path(f'smiley26_solved_upsampled_{code}.png'))

        # Method 4: simulate what Method 2 _should_ look like in QGIS without the bug by doing some hideous workarounds,
        # i.e. applyi the affine transformation in the CRS rather than in the dataset.
        step6_simulated = Path(f'smiley27_simulated_{code}.tif')
        shutil.copy(step2_georef, step6_simulated)
        orthographic_crs_ul_corner = pyproj.CRS(f'+proj=ortho +datum=WGS84 +lat_0={lat} +lon_0={lng}')
        simulated_src_and_dst_points = raster.src_and_dst_points(1000, orthographic_crs_ul_corner)
        simulated_geotransform = Geotransform.from_points(*simulated_src_and_dst_points)
        simulated_geotransform.report_error(*simulated_src_and_dst_points)
        simulated_wkt = Path('orthographic_template.wkt').read_text()
        simulated_wkt = simulated_wkt.replace('ORIGIN_LNG', str(lng))
        simulated_wkt = simulated_wkt.replace('ORIGIN_LAT', str(lat))
        simulated_wkt = simulated_wkt.replace(
            'PROJ_PIPELINE',
            (
                '+proj=pipeline +step +proj=affine '
                f'+s11={simulated_geotransform.pixel_width} '
                f'+s12={simulated_geotransform.row_rotation} '
                f'+s21={simulated_geotransform.col_rotation * -1} '
                f'+s22={simulated_geotransform.pixel_height * -1} '
            ),
        )
        step6_simulated.with_suffix('.wkt').write_text(simulated_wkt)
        simulated_geotransform.ul_x = 0.0
        simulated_geotransform.pixel_width = pixel_size**2  # don't know why this is needed
        simulated_geotransform.row_rotation = 0.0
        simulated_geotransform.ul_y = 0.0
        simulated_geotransform.col_rotation = 0.0
        simulated_geotransform.pixel_height = pixel_size**2 * -1  # don't know why this is needed
        simulated_geotransform.apply_to_dataset(step6_simulated, simulated_wkt)

        # Method 5: solve for geotransform _without_ rotation and apply to raster; this does display properly in QGIS
        step7_solved_no_rotation = Path(f'smiley28_solved_no_rotation{code}.tif')
        shutil.copy(step2_georef, step7_solved_no_rotation)
        crs = pyproj.CRS(f'epsg:{code}')
        src_pts, dst_pts = raster.src_and_dst_points(1000, crs)
        solved_geotransform = Geotransform.from_points(src_pts, dst_pts, allow_rotation=False)
        solved_geotransform.report_error(src_pts, dst_pts)
        solved_geotransform.apply_to_dataset(step7_solved_no_rotation, crs.to_wkt())
