import numpy as np
import xarray as xr
import glob
import pandas as pd
import cartopy  # Map projections libary
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import cartopy.crs as ccrs
from matplotlib.patches import Patch

############ load data from each year, combine into one dataset #################

files = sorted(glob.glob(r"C:\Users\saioa.DESKTOP-NE4E1ID\dissertation\lc_data\*.nc"))
ds = xr.open_mfdataset(files, combine="by_coords", engine="netcdf4")
print(ds["time"].values)
lc = ds["lccs_class"].load()


################ create class mappings for visualisations #######################
classes = {
    10: ("Cropland, rainfed", "#ffff64"),
    30: ("Mosaic cropland", "#dcf064"),
    40: ("Mosaic natural vegetation", "#c8c864"),
    60: ("Broadleaved deciduous forest", "#00a000"),
    70: ("Needleleaved evergreen forest", "#006400"),
    90: ("Mixed forest", "#003c00"),
    100: ("Mosaic tree and shrub", "#788200"),
    110: ("Mosaic herbaceous", "#8ca000"),
    120: ("Shrubland", "#966400"),
    130: ("Grassland", "#ffb432"),
    140: ("Lichens and mosses", "#ffdcd2"),
    150: ("Sparse vegetation", "#ffebaf"),
    160: ("Flooded forest (fresh)", "#00785a"),
    180: ("Flooded shrub/herbaceous", "#009678"),
    190: ("Urban", "#c31400"),
    200: ("Bare areas", "#fff5d7"),
    210: ("Water", "#0046c8"),
    220: ("Snow/ice", "#ffffff"),
}

# merge sub-classes
parent = {11: 10, 61: 60, 71: 70, 72: 70, 201: 200}
ids = list(classes)
cmap = ListedColormap([classes[i][1] for i in ids])

# creating a lookup table
# converting raw class value to index {0,...,N-1}
# 255 is unmapped (why?)
lut = np.full(256, 255, dtype=np.uint8)
for raw in range(256):
    p = parent.get(raw, raw)
    if p in ids:
        lut[raw] = ids.index(p)

# create index with shape (time, lat, lon)
idx = lut[lc.values]
# masks unmapped / nodata                                   
idx = np.ma.masked_equal(idx, 255)                     
years = [str(t)[:4] for t in lc["time"].values]
extent = [float(lc.lon.min()), float(lc.lon.max()),
          float(lc.lat.min()), float(lc.lat.max())]
handles = [Patch(color=classes[i][1], label=classes[i][0]) for i in ids]


############# grid of visualisations over the 11 years #################

fig, axes = plt.subplots(3, 4, figsize=(16, 9), sharex=True, sharey=True)
for ax, arr, yr in zip(axes.flat, idx, years):
    ax.imshow(arr, cmap=cmap, vmin=-0.5, vmax=len(ids) - 0.5,
              extent=extent, interpolation="nearest", aspect="auto")
    ax.set_title(yr)
# hide the unused 12th panel
for ax in axes.flat[len(years):]:
    ax.axis("off")                                     
fig.legend(handles=handles, loc="center right", fontsize=8)
fig.tight_layout(rect=[0, 0, 0.85, 1])
fig.savefig("landcover_grid.png", dpi=150)