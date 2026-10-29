import glob
import os
import pandas as pd
import rioxarray
import xarray as xr
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FOLDER = os.path.join(HERE, "composites")   # "composites" for the real run
BANDS = ["B02", "B03", "B04", "B8A", "B11", "B12"]   # same order as compositing script


def load_all():
    files = sorted(glob.glob(os.path.join(FOLDER, "median_*.tif")))
    times = [pd.Timestamp(os.path.basename(f)[7:14] + "-01") for f in files]   # median_YYYY-MM.tif
    arrays = [rioxarray.open_rasterio(f, masked=True, chunks=True) for f in files]
    da = xr.concat(arrays, dim=pd.Index(times, name="time"))
    return da.assign_coords(band=BANDS)       # select by name from now on


def norm_diff(a, b, name, long_name):
    d = a + b
    out = ((a - b) / d.where(d != 0)).clip(-1, 1)
    out.name = name
    out.attrs.update(long_name=long_name, units="1", valid_range=[-1.0, 1.0])
    return out


def plot_index(da, cmap="RdYlGn", ncols=4):
    g = da.plot.imshow(col="time", col_wrap=ncols, cmap=cmap, vmin=-1, vmax=1,
                       cbar_kwargs={"label": da.name.upper()})
    g.set_titles("{value:%Y-%m}")
    for ax in g.axs.flat:
        ax.axis("off")
    out = os.path.join(HERE, f"{da.name}.png")
    g.fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved {out}")
    plt.show()


if __name__ == "__main__":
    med = load_all()
    ndvi = norm_diff(med.sel(band="B8A"), med.sel(band="B04"), "ndvi",
                     "Normalized Difference Vegetation Index").compute()
    nbr = norm_diff(med.sel(band="B8A"), med.sel(band="B12"), "nbr",
                    "Normalized Burn Ratio").compute()

    aug_months = ["2015-08-01", "2016-08-01", "2017-08-01", 
                  "2018-08-01", "2019-08-01", "2020-08-01",
                  "2021-08-01", "2022-08-01"]
    ndvi_sel = ndvi.sel(time=aug_months)
    plot_index(ndvi_sel, ncols=2)
    nbr_sel = nbr.sel(time = aug_months)
    plot_index(nbr_sel, ncols = 2)

    # Burn severity: pre-fire minus post-fire NBR (the Kenow fire started in Sept 2017)
    dnbr = nbr.sel(time="2016-08").squeeze() - nbr.sel(time="2022-08").squeeze()