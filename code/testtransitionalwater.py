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
from lxml import etree
from matplotlib.animation import FuncAnimation
import folium
from folium.plugins import TimestampedGeoJson
import json
import matplotlib.animation as animation
import ipywidgets as widgets
from ipywidgets import interact

# %% --------------------------------------------------------------------------
# Loading Data sets
# -----------------------------------------------------------------------------
# Path to your GML file
gml_file = r'C:\Users\Dpayn\MLE\Machine-Learning\data\water\TransitionalWaterBodies-1st-cycle.gml'

# Read the GML file into a GeoDataFrame
gdf = gpd.read_file(gml_file)

# Print the first few rows of the GeoDataFrame
print(gdf.head())

# Plot the GeoDataFrame
gdf.plot()
plt.show()



# %%
# Path to your GML file
gml_file = r'C:\Users\Dpayn\MLE\Machine-Learning\data\water\TransitionalWaterBodies-1st-cycle.gml'

# Parse the GML file
tree = etree.parse(gml_file)
root = tree.getroot()

# Print the root element and namespaces
print(f"Root element: {root.tag}")
print(f"Namespaces: {root.nsmap}")

# Extract specific elements (example: all the feature members)
for feature_member in root.findall('.//{http://www.opengis.net/gml}featureMember'):
    print(etree.tostring(feature_member, pretty_print=True).decode())



# %%
gdf = gpd.read_file(gml_file)

# Display the first few rows of the GeoDataFrame
print(gdf.head())

# Plot the GeoDataFrame with a basemap
ax = gdf.plot(figsize=(10, 10), alpha=0.5, edgecolor='k')
ctx.add_basemap(ax, crs=gdf.crs.to_string())

# Add a title to the plot
plt.title('Transitional Water Bodies - 1st Cycle')
plt.show()


# %%
# Read the GML file into a GeoDataFrame
gdf = gpd.read_file(gml_file)

# Ensure the data is in Web Mercator for contextily
gdf = gdf.to_crs(epsg=3857)

# Extract time data from your dataset (assuming there is a time-related column, e.g., 'date')
# For demonstration, let's create a mock 'date' column
import pandas as pd
gdf['date'] = pd.date_range(start='2020-01-01', periods=len(gdf), freq='M')

# Sort by date
gdf = gdf.sort_values('date')

# Create a figure and axis
fig, ax = plt.subplots(figsize=(10, 10))

def update(frame):
    ax.clear()
    current_date = gdf.iloc[frame]['date']
    subset = gdf[gdf['date'] <= current_date]
    subset.plot(ax=ax, alpha=0.5, edgecolor='k')
    ctx.add_basemap(ax, crs=subset.crs.to_string())
    ax.set_title(f'Transitional Water Bodies - {current_date.date()}')

# Create the animation
anim = FuncAnimation(fig, update, frames=len(gdf), repeat=False)

# Save the animation as a GIF
anim.save('transitional_water_bodies.gif', writer='pillow')

plt.show()



# %%

# Ensure the data is in WGS84 (lat/lon) for Folium
gdf = gdf.to_crs(epsg=4326)

# Check the length of the column and get the first and last timestamps
length = len(gdf['INS_WHEN'])
first_timestamp = gdf['INS_WHEN'].min()
last_timestamp = gdf['INS_WHEN'].max()

print("Length of the INS_WHEN column:", length)
print("First Timestamp:", first_timestamp)
print("Last Timestamp:", last_timestamp)

# Define the function to create GeoJSON features
def create_geojson_features(gdf):
    features = []
    for _, row in gdf.iterrows():
        if row['geometry'].geom_type == 'Polygon':
            geometry = json.loads(row['geometry'].to_json())
        elif row['geometry'].geom_type == 'MultiPolygon':
            # If MultiPolygon, convert each Polygon separately
            polygons = []
            for polygon in row['geometry'].geoms:  # Accessing individual polygons in MultiPolygon
                polygon_geojson = {
                    'type': 'Polygon',
                    'coordinates': [list(polygon.exterior.coords)]
                }
                polygons.append(polygon_geojson)
            geometry = {
                'type': 'MultiPolygon',
                'coordinates': polygons
            }
        else:
            # Handle other geometry types if needed
            continue

        feature = {
            'type': 'Feature',
            'geometry': geometry,
            'properties': {
                'time': row['INS_WHEN'].strftime('%Y-%m-%d'),
                'popup': row['NAME'] if 'NAME' in row else 'N/A'
            }
        }
        features.append(feature)
    return features

# Create GeoJSON features
features = create_geojson_features(gdf)

# Construct GeoJSON data
geojson_data = {
    'type': 'FeatureCollection',
    'features': features
}

# Create a base map
m = folium.Map(location=[54.5, -6.5], zoom_start=8)

# Add time slider using TimestampedGeoJson
TimestampedGeoJson(
    geojson_data,
    transition_time=200,
    period='P1M',  # Period of time (1 month in this case)
    add_last_point=True,
    auto_play=True,
    loop=True,
    date_options='YYYY-MM-DD',
    time_slider_drag_update=True
).add_to(m)

# Save the map to an HTML file
m.save('transitional_water_bodies.html')

# Display the map
m


# %%
# Read the GML file into a GeoDataFrame
gdf = gpd.read_file(gml_file)

# Ensure the data is in WGS84 (lat/lon)
gdf = gdf.to_crs(epsg=4326)

# Sort the GeoDataFrame by the timestamp column
gdf = gdf.sort_values(by='INS_WHEN')

# Create a figure and axis
fig, ax = plt.subplots()

# Initialize an empty plot
plot = ax.plot([], [], color='blue')[0]

# Function to update the plot for each timestamp
def update_plot(timestamp):
    # Clear the axis
    ax.clear()
    # Plot the polygons for the current timestamp
    gdf[gdf['INS_WHEN'] == timestamp].plot(ax=ax, color='blue', edgecolor='black')
    # Set axis labels
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    # Set title
    ax.set_title('Transitional Water Bodies on ' + str(timestamp.date()))

# Get the timestamps
timestamps = gdf['INS_WHEN'].unique()

# Animate the plot
ani = FuncAnimation(fig, update_plot, frames=timestamps, repeat=False)

# Show the animation
plt.show()

# %%
# Sort the GeoDataFrame by the INS_WHEN column
gdf_sorted = gdf.sort_values(by='INS_WHEN')

def plot_polygon(ax, polygon):
    x, y = polygon.exterior.xy
    ax.plot(x, y, color='black')

def update_plot(frame):
    plt.clf()  # Clear the plot
    gdf_frame = gdf_sorted.iloc[[frame]]
    ax = plt.gca()
    geom = gdf_frame.geometry.item()  # Extract the geometry object from the Series
    if geom.geom_type == 'Polygon':
        plot_polygon(ax, geom)
    elif geom.geom_type == 'MultiPolygon':
        for polygon in geom.geoms:  # Use .geoms to iterate over the polygons in a MultiPolygon
            plot_polygon(ax, polygon)
    ax.set_xlim(gdf.total_bounds[0], gdf.total_bounds[2])  # Set x-axis limits
    ax.set_ylim(gdf.total_bounds[1], gdf.total_bounds[3])  # Set y-axis limits
    
    # Add a text box to the plot
    date = gdf_frame['INS_WHEN'].item()
    plt.text(0.05, 0.95, str(date), transform=ax.transAxes, verticalalignment='top')
    
    plt.show()

# Create a slider to interactively select the frame
interact(update_plot, frame=(0, len(gdf_sorted)-1))
# %%
