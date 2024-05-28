""
#Testing all water datasets I can access


#__date__ = "27-05-24"
#__author__ = "WaveyDavey"


""

# %%
# Imports
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
from lxml import etree
from matplotlib.animation import FuncAnimation
import folium
from folium.plugins import TimestampedGeoJson
import json
import matplotlib.animation as animation
import ipywidgets as widgets
from ipywidgets import interact
import xarray as xr
import os
import matplotlib.colors as mcolors
from IPython.display import display
#import pygrib

# %%
# Loading Data sets
# Correct path to your GML file
gml_file = 'C:\\Users\\Dpayn\\MLE\\Machine-Learning\\data\\water\\RiverSegment.gml'

# Read the GML file into a GeoDataFrame
gdf = gpd.read_file(gml_file)

# Define a color map for the segment types
color_map = {
    'Y': 'blue',  # real segment
    'N': 'green',  # real underground segment
    'L': 'red',  # virtual lake segment
    'T': 'purple',  # virtual transitional water segment
    'C': 'orange',  # virtual coastal water segment
    'V': 'gray'  # unclassified virtual segment
}

# Create a new column 'color' that maps the segment types to colors
gdf['color'] = gdf['SEGMENT_TYPE'].map(color_map)

# Fill NaN values with a default color
gdf['color'] = gdf['color'].fillna('black')

# Create a function to plot the data
def plot_data(segment_types):
    fig, ax = plt.subplots(figsize=(10, 10))
    gdf[gdf['SEGMENT_TYPE'].isin(segment_types)].to_crs(epsg=3857).plot(ax=ax, alpha=0.5, color=gdf['color'])  # Convert to Web Mercator (EPSG:3857)
    try:
        ctx.add_basemap(ax, zoom=10)  # Add basemap from OpenStreetMap
    except TypeError as e:
        print(f"Failed to add basemap: {e}")
    plt.show()

# Create a multi-select widget for the segment types
segment_types_widget = widgets.SelectMultiple(
    options=color_map.keys(),
    value=list(color_map.keys()),
    description='Segment Types',
    disabled=False
)

# Display the widget
display(segment_types_widget)

# Call the plot function when the widget value changes
widgets.interactive_output(plot_data, {'segment_types': segment_types_widget})
# %%
