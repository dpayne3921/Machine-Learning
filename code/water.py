# %% 


import geopandas as gpd
import pandas as pd
# Load GeoJSON file into a GeoDataFrame
data_path = r'C:\Users\Dpayn\MLE\Machine-Learning\data\Northern_Ireland_Groundwater_Bodies.geojson'
gdf = gpd.read_file(data_path)

# Explore the GeoDataFrame
print(gdf.head())

# %%
# Feature Engineering: Adding Centroid Coordinates
gdf['centroid'] = gdf.geometry.centroid
gdf['centroid_x'] = gdf.centroid.x
gdf['centroid_y'] = gdf.centroid.y

# Feature Engineering: Adding Bounding Box dimensions
gdf['bbox_area'] = gdf.geometry.envelope.area
gdf['bbox_min_x'] = gdf.geometry.bounds.minx
gdf['bbox_min_y'] = gdf.geometry.bounds.miny
gdf['bbox_max_x'] = gdf.geometry.bounds.maxx
gdf['bbox_max_y'] = gdf.geometry.bounds.maxy

# Drop geometry and unnecessary columns for machine learning
features = gdf.drop(columns=['OBJECTID', 'NAME', 'GlobalID', 'geometry', 'centroid'])

# Ensure all data is numeric for machine learning
features = features.apply(pd.to_numeric, errors='coerce')
features = features.dropna()

print(features.head())
# %%
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Assume we want to cluster the data into 3 groups
kmeans = KMeans(n_clusters=3)
gdf['cluster'] = kmeans.fit_predict(features)

# Visualize the clusters
gdf.plot(column='cluster', categorical=True, legend=True, figsize=(10, 6))
plt.show()
# %%
import geopandas as gpd
import contextily as ctx
import matplotlib.pyplot as plt

# Load GeoJSON file into a GeoDataFrame
data_path = r'C:\Users\Dpayn\MLE\Machine-Learning\data\Northern_Ireland_Groundwater_Bodies.geojson'
gdf = gpd.read_file(data_path)

# Project to Web Mercator for compatibility with tile maps
gdf = gdf.to_crs(epsg=3857)

# Plot with contextily basemap
ax = gdf.plot(figsize=(10, 10), alpha=0.5, edgecolor='k')
ctx.add_basemap(ax)
plt.show()
# %%
import geopandas as gpd
import matplotlib.pyplot as plt

# Load GeoJSON file into a GeoDataFrame
data_path = r'C:\Users\Dpayn\MLE\Machine-Learning\data\Northern_Ireland_Groundwater_Bodies.geojson'
gdf = gpd.read_file(data_path)

# Plot the GeoDataFrame
gdf.plot()
plt.show()
# %%
import geopandas as gpd
import plotly.express as px

# Load GeoJSON file into a GeoDataFrame
data_path = r'C:\Users\Dpayn\MLE\Machine-Learning\data\Northern_Ireland_Groundwater_Bodies.geojson'
gdf = gpd.read_file(data_path)

# Plot using Plotly
fig = px.choropleth(gdf, geojson=gdf.geometry, locations=gdf.index, color="Shape__Area",
                    color_continuous_scale="Viridis",
                    projection="mercator", title="Groundwater Body Areas")
fig.update_geos(fitbounds="locations")
fig.show()
# %%
