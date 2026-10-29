import rioxarray
from rioxarray.merge import merge_arrays
from rasterio.enums import Resampling

FIRE_AREA = [-114.4, 48.9, -113.7, 49.2]          # [W, S, E, N]
BASE = "https://copernicus-dem-30m.s3.amazonaws.com"
TILES = [("N48", "W115"), ("N48", "W114"), ("N49", "W115"), ("N49", "W114")]


def tile_url(lat, lon):
    name = f"Copernicus_DSM_COG_10_{lat}_00_{lon}_00_DEM"
    return f"{BASE}/{name}/{name}.tif"


tiles = [rioxarray.open_rasterio(tile_url(*t), masked=True).squeeze() for t in TILES]
dem = merge_arrays(tiles)

# clip with a small buffer, in lat/lon, before reprojecting (edges get interpolated)
w, s, e, n = FIRE_AREA
dem = dem.rio.clip_box(w - 0.02, s - 0.02, e + 0.02, n + 0.02)

# resample straight onto your Sentinel-2 grid: same CRS, extent, 20 m pixels
ref = rioxarray.open_rasterio("composites/median_2017-08.tif")
dem20 = dem.rio.reproject_match(ref, resampling=Resampling.bilinear)

dem20.rio.to_raster("dem_20m.tif")