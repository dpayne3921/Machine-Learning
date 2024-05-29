# %%
import geopandas as gpd
import folium
import pandas as pd

#%%

# Define the file path
file_path = r"C:\Users\Dpayn\MLE\Machine-Learning\Large_data\River_Water_Quality_Monitoring_1990_to_2018_-_pH.geojson"

# Read the file
gdf = gpd.read_file(file_path)

# Convert datetime values to strings
gdf = gdf.astype({col: str for col in gdf.select_dtypes(include=[pd.Timestamp]).columns})

# Create a map centered at the mean of latitude and longitude
m = folium.Map(location=[gdf.geometry.centroid.y.mean(), gdf.geometry.centroid.x.mean()], zoom_start=10)

# Add the GeoDataFrame to the map
folium.GeoJson(gdf).add_to(m)

# Display the map
m
# %%
