"""
Sanity checks for the composites written by s2_loading_stac_alldata.py.

Reads the files in ./composites_test/ (TEST_MODE output), prints a PASS / WARN /
FAIL line per check, and saves a PNG per month with true colour, false colour
and the clear-observation count.

    python check_composites.py
"""
import json
import os
import matplotlib.pyplot as plt
import numpy as np
import rioxarray

HERE = os.path.dirname(os.path.abspath(__file__))
FOLDER = os.path.join(HERE, "composites_test")   # use "composites" for the real run
MONTHS = ["2017-08", "2022-08"]                            # add e.g. "2022-08" to compare years
BANDS = ["B02", "B03", "B04", "B8A", "B11", "B12"]
EPSG = 32611


def report(status, msg):
    print(f"  [{status}] {msg}")


def check(month):
    print(f"\n=== {month} ===")
    med = rioxarray.open_rasterio(os.path.join(FOLDER, f"median_{month}.tif"), masked=True)
    ncl = rioxarray.open_rasterio(os.path.join(FOLDER, f"nclear_{month}.tif"), masked=True).squeeze()
    with open(os.path.join(HERE, "items", f"{month}.json")) as f:
        items = json.load(f)
    n_dates = len({it["properties"]["datetime"][:10] for it in items})

    # --- structure
    ok = med.sizes["band"] == len(BANDS)
    report("PASS" if ok else "FAIL", f"{med.sizes['band']} bands (expected {len(BANDS)})")
    crs = med.rio.crs.to_epsg()
    report("PASS" if crs == EPSG else "FAIL", f"CRS EPSG:{crs}")
    same = med.shape[1:] == ncl.shape
    report("PASS" if same else "FAIL", f"median grid {med.shape[1:]} vs nclear grid {ncl.shape}")

    # --- value range (reflectance should sit roughly between 0 and 1)
    lo, hi = float(med.min()), float(np.nanpercentile(med.values, 99.9))
    report("PASS" if lo >= 0 and hi <= 1.2 else "WARN",
           f"reflectance min {lo:.3f}, 99.9th percentile {hi:.3f}")
    means = {b: float(med.sel(band=i + 1).mean()) for i, b in enumerate(BANDS)}
    print("  band means: " + ", ".join(f"{b} {v:.3f}" for b, v in means.items()))
    # vegetation: near infrared above red; a flat or reversed pattern is suspicious
    report("PASS" if means["B8A"] > means["B04"] else "WARN",
           "NIR (B8A) above red (B04) on average, as expected for vegetated land")

    # --- missing data
    nan_frac = float(med.sel(band=1).isnull().mean())
    report("PASS" if nan_frac < 0.2 else "WARN", f"{nan_frac:.1%} of pixels have no value")
    zero_clear = ncl.fillna(0) == 0
    consistent = bool((zero_clear == med.sel(band=1).isnull()).all())
    report("PASS" if consistent else "FAIL", "pixels with 0 clear observations are exactly the NaN pixels")

    # --- clear observation counts
    nmax = int(ncl.max())
    report("PASS" if nmax <= len(items) else "FAIL",
           f"max clear observations {nmax} <= {len(items)} items ({n_dates} distinct dates)")
    print(f"  clear observations per pixel: mean {float(ncl.mean()):.1f}, "
          f"median {float(ncl.median()):.0f}")

    # --- pictures
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    med.sel(band=[3, 2, 1]).plot.imshow(rgb="band", robust=True, ax=axes[0])
    axes[0].set_title("True colour (B04, B03, B02)")
    med.sel(band=[6, 4, 3]).plot.imshow(rgb="band", robust=True, ax=axes[1])
    axes[1].set_title("False colour (B12, B8A, B04)")
    ncl.plot.imshow(ax=axes[2], cmap="viridis", cbar_kwargs={"label": "clear observations"})
    axes[2].set_title("Clear observations per pixel")
    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel(""); ax.set_ylabel("")
    fig.suptitle(month)
    fig.tight_layout()
    out = os.path.join(HERE, f"check_{month}.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  saved {out}")
    return means


if __name__ == "__main__":
    results = {m: check(m) for m in MONTHS}
    if len(results) > 1:
        print("\n=== band means across months (for the offset check) ===")
        for m, means in results.items():
            print(f"  {m}: " + ", ".join(f"{b} {v:.3f}" for b, v in means.items()))