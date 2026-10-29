"""
Monthly cloud-masked median composites of Sentinel-2 L2A over the Kenow
wildfire area, 2015-2022, from the Copernicus Data Space STAC catalogue.

For each month it saves, in ./composites/:
    median_YYYY-MM.tif   median reflectance, one band per entry in BANDS (0-1)
    nclear_YYYY-MM.tif   number of clear (unmasked) observations per pixel

Safe to stop and restart: finished months are skipped, and each month's
STAC search is cached in ./items/.

Run from a terminal with the myenv environment active:
    python monthly_composites.py
"""
import calendar
import json
import os
import re
import time
import traceback

import dask
import numpy as np
import pystac_client
import rioxarray  # noqa: F401  (registers the .rio accessor)
import stackstac
import xarray as xr
from dask.diagnostics import ProgressBar
from dotenv import load_dotenv
from odc.algo import mask_cleanup
from pystac_client.stac_api_io import StacApiIO
from tqdm import tqdm
from urllib3 import Retry

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
FIRE_AREA = [-114.4, 48.9, -113.7, 49.2]          # [W, S, E, N]
EPSG = 32611
RESOLUTION = 20                                    # metres
BANDS = ["B02_20m", "B03_20m", "B04_20m", "B8A_20m", "B11_20m", "B12_20m"]
SCL = "SCL_20m"

YEARS = range(2015, 2023)
MONTHS = range(1, 13)

# SCL classes treated as unusable:
# 0 no data, 1 saturated, 3 cloud shadow, 8/9 cloud, 10 thin cirrus
BAD_SCL = [0, 1, 3, 8, 9, 10]

NUM_WORKERS = 4        # download threads; lower to 1 if the kernel crashes
CHUNK = 1024           # spatial chunk size (pixels) when building the stack
MAX_ITEMS = 300        # per-month cap on the STAC search

# Test mode: only process these months, on a small window, and write to
# separate folders so test output never mixes with the real run.
TEST_MODE = False
TEST_MONTHS = [(2017, 8), (2022, 8)]
# TEST_WINDOW = dict(x=slice(1000, 1500), y=slice(800, 1300))   # pixels
TEST_WINDOW = None

# ----------------------------------------------------------------------------
# Setup
# ----------------------------------------------------------------------------
load_dotenv()
os.environ["AWS_ACCESS_KEY_ID"] = os.environ["CDSE_S3_ACCESS_KEY"]
os.environ["AWS_SECRET_ACCESS_KEY"] = os.environ["CDSE_S3_SECRET_KEY"]
os.environ["AWS_S3_ENDPOINT"] = "eodata.dataspace.copernicus.eu"
os.environ["AWS_HTTPS"] = "YES"
os.environ["AWS_VIRTUAL_HOSTING"] = "FALSE"
os.environ["GDAL_HTTP_UNSAFESSL"] = "YES"
os.environ["GDAL_HTTP_TCP_KEEPALIVE"] = "YES"

HERE = os.path.dirname(os.path.abspath(__file__))
SUFFIX = "_test" if TEST_MODE else ""
ITEMS_DIR = os.path.join(HERE, "items")
OUT_DIR = os.path.join(HERE, "composites" + SUFFIX)
LOG_FILE = os.path.join(HERE, f"progress{SUFFIX}.log")
os.makedirs(ITEMS_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

retry = Retry(
    total=6,
    backoff_factor=2,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=None,
    respect_retry_after_header=True,
)
cat = pystac_client.Client.open(
    "https://stac.dataspace.copernicus.eu/v1",
    stac_io=StacApiIO(max_retries=retry),
)
cat.add_conforms_to("ITEM_SEARCH")


def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
    tqdm.write(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def get_items(year, month):
    """STAC items for one month, cached on disk so searches run only once."""
    cache = os.path.join(ITEMS_DIR, f"{year}-{month:02d}.json")
    if os.path.exists(cache):
        with open(cache) as f:
            return json.load(f)

    last_day = calendar.monthrange(year, month)[1]
    search = cat.search(
        collections=["sentinel-2-l2a"],
        bbox=FIRE_AREA,
        datetime=f"{year}-{month:02d}-01T00:00:00Z/{year}-{month:02d}-{last_day}T23:59:59Z",
        sortby=[{"field": "properties.datetime", "direction": "asc"}],
        max_items=MAX_ITEMS,
    )
    items = list(search.items_as_dicts())
    with open(cache, "w") as f:
        json.dump(items, f)
    return items


def boa_offset(item_id):
    """
    Reflectance offset for a Sentinel-2 L2A product, from the processing
    baseline in its name (e.g. ..._N0500_...). Baseline 04.00 and later add
    +1000 to the digital numbers, earlier baselines do not.
    """
    m = re.search(r"_N(\d{4})_", item_id)
    if m and int(m.group(1)) >= 400:
        return 1000.0
    return 0.0


def composite_month(year, month):
    out_median = os.path.join(OUT_DIR, f"median_{year}-{month:02d}.tif")
    out_nclear = os.path.join(OUT_DIR, f"nclear_{year}-{month:02d}.tif")
    if os.path.exists(out_median):
        return "skipped"

    items = get_items(year, month)
    if not items:
        return "no items"

    stack = stackstac.stack(
        items=items,
        assets=BANDS + [SCL],
        resolution=(RESOLUTION, RESOLUTION),
        bounds_latlon=FIRE_AREA,
        epsg=EPSG,
        chunksize=CHUNK,
        dtype="float32",
        fill_value=np.float32(np.nan),  
        rescale=False,          # raw digital numbers; scaled below
    )
    if TEST_MODE and TEST_WINDOW is not None:
        stack = stack.isel(**TEST_WINDOW)

    ds = stack.to_dataset(dim="band")

    # --- cloud mask from SCL (opening removes specks, dilation widens edges)
    bad = ds[SCL].isin(BAD_SCL)
    bad = mask_cleanup(bad, mask_filters=[("opening", 2), ("dilation", 5)])

    # --- reflectance: drop nodata (0), remove per-product offset, scale
    offsets = np.array([boa_offset(str(i)) for i in stack.id.values], dtype="float32")
    offsets = xr.DataArray(offsets, dims="time")     # aligned by position

    refl = ds[BANDS].where(~bad)
    refl = refl.where(refl > 0)
    refl = ((refl - offsets) / 10000.0).clip(min=0)

    # the median needs every date for a pixel in one chunk
    refl = refl.chunk({"time": -1, "y": 512, "x": 512})

    median = refl.median("time", skipna=True)
    n_clear = refl[BANDS[0]].notnull().sum("time")

    with dask.config.set(scheduler="threads", num_workers=NUM_WORKERS), ProgressBar():
        median, n_clear = dask.compute(median, n_clear)

    arr = median.to_array("band").rio.write_crs(f"EPSG:{EPSG}")
    arr.attrs["long_name"] = tuple(BANDS)            # band names in the file
    arr.rio.to_raster(out_median)

    n_clear.astype("uint16").rio.write_crs(f"EPSG:{EPSG}").rio.to_raster(out_nclear)
    return f"{len(items)} items"


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    if TEST_MODE:
        jobs = list(TEST_MONTHS)
    else:
        jobs = [(y, m) for y in YEARS for m in MONTHS]

    todo = [
        (y, m)
        for y, m in jobs
        if not os.path.exists(os.path.join(OUT_DIR, f"median_{y}-{m:02d}.tif"))
    ]
    log(f"{len(todo)} months to do ({len(jobs) - len(todo)} already done)"
        f"{' [TEST MODE]' if TEST_MODE else ''}")

    failed = []
    t_start = time.time()
    bar = tqdm(todo, unit="month", desc="Composites")
    for y, m in bar:
        bar.set_postfix_str(f"{y}-{m:02d}")
        t0 = time.time()
        try:
            result = composite_month(y, m)
            log(f"{y}-{m:02d} {result}, {time.time() - t0:.0f} s")
        except Exception:
            failed.append((y, m))
            log(f"{y}-{m:02d} FAILED:\n{traceback.format_exc()}")

    log(f"finished in {(time.time() - t_start) / 60:.0f} min; failed months: {failed}")