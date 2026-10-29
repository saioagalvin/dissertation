import glob
import os
import pandas as pd
import rioxarray
import xarray as xr
import matplotlib.pyplot as plt

class data_handling:

    def __init__(self, filepath, s2_composites_folder, s2_bands):
        self.filepath = filepath
        self.s2_composites_folder = s2_composites_folder
        self.s2_bands = s2_bands


    def load_s2_composites(self):
        '''
        Load geoTIFF files from the folder that contains the median composites.
        '''

        files = sorted(glob.glob(os.path.join(self.s2_composites_folder, "median_*.tif")))

        # extract time from filename of format median_YYYY-MM.tif
        times = [pd.Timestamp(os.path.basename(f)[7:14] + "-01") for f in files]

        arrays = [rioxarray.open_rasterio(f, masked=True, chunks=True) for f in files]
        da = xr.concat(arrays, dim=pd.Index(times, name="time"))

        # select bands by name instead of index
        da_concat = da.assign_coords(band = self.s2_bands)
        self.loaded_composites = da_concat

        return da_concat


    def norm_diff(self, a, b, name, long_name):
        '''
        Calculate normalised difference between two values. Useful for NDVI and NBR calculation.
        '''

        d = a + b
        result = ((a - b) / d.where(d != 0)).clip(-1, 1)
        result.name = name
        result.attrs.update(long_name=long_name, units="1", valid_range=[-1.0, 1.0])
        self.norm_diff = result

        return result


    def plot_index(self, data, cmap="RdYlGn", ncols=4, value_range, save = False):
        '''
        Plot data for a specific index, with each subplot showing a month.'''
        g = data.plot.imshow(col="time", 
                             col_wrap=ncols, 
                             cmap=cmap, 
                             vmin=value_range[0], 
                             vmax=value_range[1],
                             cbar_kwargs={"label": data.name.upper()})
        g.set_titles("{value:%Y-%m}")

        for ax in g.axs.flat:
            ax.axis("off")

        if save == True:
            png_name = os.path.join(self.filepath, f"{data.name}.png")
            g.fig.savefig(png_name, dpi=120, bbox_inches="tight")
            print(f"saved {png_name}")

        plt.show()