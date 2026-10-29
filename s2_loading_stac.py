import pystac_client
import geogif
import stackstac
import os
import json
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import numpy as np
import sys
import rioxarray
from odc.algo import mask_cleanup
from odc.algo import geomedian_with_mads
from dask.diagnostics import ProgressBar

#### S3 credentials
load_dotenv() 
os.environ["AWS_ACCESS_KEY_ID"] = os.environ["CDSE_S3_ACCESS_KEY"]
os.environ["AWS_SECRET_ACCESS_KEY"] = os.environ["CDSE_S3_SECRET_KEY"]
os.environ["GDAL_HTTP_TCP_KEEPALIVE"] = "YES"
os.environ["AWS_S3_ENDPOINT"] = "eodata.dataspace.copernicus.eu"
os.environ["AWS_HTTPS"] = "YES"
os.environ["AWS_VIRTUAL_HOSTING"] = "FALSE"
os.environ["GDAL_HTTP_UNSAFESSL"] = "YES"


#### constants
HERE = os.path.dirname(os.path.abspath(__file__))
FIRE_AREA = [-114.4, 48.9, -113.7, 49.2]
BANDS = ["B02_20m", "B03_20m", "B04_20m", "B8A_20m", "B11_20m", "B12_20m"]       # B08_10m is NIR 1 but want consistency of 20m
# 0 no data, 1 saturated, 3 cloud shadow, 8/9 cloud medium/high, 10 thin cirrus
SCL_BANDS = [0, 1, 3, 8, 9, 10]

#### Query

# We query the catalog and convert the results into a list of dictionaries
# using the stac query run from search_items.py and saved in kenow_items.json
with open("kenow_items.json") as f:
    items = json.load(f)


#### Create a datacube

# After performing the search, we can create a STACK object from the obtained items. In the definition of the STACK object, the following parameters must be declared:

# items: The STAC items that will be included in the stack object.
# resolution: The desired output resolution.
# bounds_latlon: The spatial bounding box of the output, given in degrees of latitude and longitude.
# chunksize: The chunk size to be used for the Dask array.
# epsg: The EPSG code for the coordinate reference system to which the data will be reprojected.
# gdal_env: GDAL configuration options to be used when opening and reading the datasets.

stack = stackstac.stack(
    items = items,
    assets = BANDS + ["SCL_20m"], # include scl here
    resolution = (20, 20),  # SCL, B8A, B11 and B12 are natively 20m but 10m bands get resampled down so consider (10,10) at cost of more data
    bounds_latlon = FIRE_AREA,
    chunksize=1024,
    epsg=32611,
    gdal_env=stackstac.DEFAULT_GDAL_ENV.updated(
        {
            "GDAL_NUM_THREADS": -1,
            "GDAL_HTTP_UNSAFESSL": "YES",
            "GDAL_HTTP_TCP_KEEPALIVE": "YES",
            "AWS_VIRTUAL_HOSTING": "FALSE",
            "AWS_HTTPS": "YES",
        }
    ),
)


# make band labels into variable names
# easier to work with for masking
ds = stack.to_dataset(dim="band") 


#### masking using SCL

scl = ds["SCL_20m"]

# remove isolated specks, then grow the mask to catch cloud edges
mask_to_remove = mask_cleanup(scl.isin(SCL_BANDS), mask_filters=[("opening", 2), ("dilation", 5)])

refl = ds[BANDS].where(~mask_to_remove)
refl = refl.where(refl > 0)              # zeros represent nodata
refl = refl / 10000.0                    # 2017 data has no +1000 offset, this only started in 2022
refl = refl.chunk({"time": -1, "y": 256, "x": 256})   # geomedian needs all dates in one chunk

#### create geomedian composite
# comp = geomedian_with_mads(refl, work_chunks=(128, 128)).compute(scheduler="single-threaded")
# or otherwise just median:
with ProgressBar():
    comp = refl.groupby("time.month").median("time", skipna=True).compute(scheduler="single-threaded")
print(comp)

out = os.path.join(HERE, "composite_aug_oct17.tif")
arr = comp[BANDS].to_array("band")
arr = arr.rio.write_crs("EPSG:32611")
arr.rio.to_raster(out)
print("saved to", out)

sys.exit()

#### Plotting rgb data
rgb = comp[["B04_20m", "B03_20m", "B02_20m"]].to_array("band")
rgb_day = (
    rgb.assign_coords(day=rgb.time.dt.floor("D"))
    .groupby("day")
    .first(skipna=True)
    .rename(day="time")
)
print(rgb_day.time.dt.strftime("%Y-%m-%d").values)
rgb_day_sel = rgb_day.sel(time="2017-08-12").squeeze()

# rgb_gif = geogif.dgif(rgb, fps=1).compute()
# img = geogif.dgif(rgb, fps=1, robust=True).compute()

# out = r"C:\Users\saioa.DESKTOP-NE4E1ID\dissertation\rgb_aug_oct17.gif"
# with open(out, "wb") as f:
#     f.write(img.data)

img = rgb_day_sel.compute(scheduler="single-threaded")
img.plot.imshow(rgb="band", robust=True, figsize=(8, 6))
plt.title("RGB of Kenow wildfire area")
plt.axis("off")
plt.show()