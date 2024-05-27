""
#Testing all water datasets I can access


#__date__ = "27-05-24"
#__author__ = "WaveyDavey"


""

# %% --------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import contextily as ctx

# %% --------------------------------------------------------------------------
# Loading Data sets
# -----------------------------------------------------------------------------
# Load GeoJSON file into a GeoDataFrame
data_path = r'C:\Users\Dpayn\MLE\Machine-Learning\data\DWPA_Abstraction_Point_Hydrological_Catchments_-6604591403590434389.geojson'
gdf = gpd.read_file(data_path)

# Explore the GeoDataFrame
#print(gdf.head())


# %%
fig, ax = plt.subplots(figsize=(10, 10))
gdf.boundary.plot(ax=ax, linewidth=1)
gdf.plot(ax=ax, alpha=0.5, edgecolor='k')

plt.title('Water Bodies and Basins')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()
# %%

# Convert to Web Mercator for contextily basemap
gdf = gdf.to_crs(epsg=3857)

# Plot with contextily basemap
ax = gdf.plot(figsize=(10, 10), alpha=0.5, edgecolor='k')
ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
plt.title('Drinking Water Protected Areas (DWPAs)')
plt.show()
# %%
