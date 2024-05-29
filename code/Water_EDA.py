# %%

import geopandas as gpd
import folium
from folium import LayerControl
from IPython.display import display, HTML



# %%
# Define the paths to the GML files
data_paths = {
    "Catchment Stakeholder Group": "C:/Users/Dpayn/MLE/Machine-Learning/Large_data/CatchmentStakeholderGroup.gml",
    "Local Management Area": "C:/Users/Dpayn/MLE/Machine-Learning/Large_data/LocalManagementArea.gml",
    "River Segment": "C:/Users/Dpayn/MLE/Machine-Learning/Large_data/RiverSegment.gml",
    "Transitional Water Bodies - 1st Cycle": "C:/Users/Dpayn/MLE/Machine-Learning/Large_data/TransitionalWaterBodies-1st-cycle.gml"
}

# Load the files into GeoDataFrames
gdfs = {name: gpd.read_file(path) for name, path in data_paths.items()}

# Inspect each GeoDataFrame
for name, gdf in gdfs.items():
    print(f"--- {name} ---")
    print(gdf.info())
    print(gdf.head())
    print("\n")


# %%
# Function to simplify geometries for better performance
def simplify_geometries(gdf, tolerance=0.001):
    gdf['geometry'] = gdf['geometry'].simplify(tolerance, preserve_topology=True)
    return gdf

# Simplify GeoDataFrames
gdfs = {name: simplify_geometries(gdf) for name, gdf in gdfs.items()}

# %%
def generic_style_function(color='blue'):
    return lambda feature: {
        'fillColor': color,
        'color': color,
        'weight': 2,
        'fillOpacity': 0.5
    }

# Example of different colors for each layer
layer_colors = {
    "Catchment Stakeholder Group": 'green',
    "Local Management Area": 'orange',
    "River Segment": 'blue',
    "Transitional Water Bodies - 1st Cycle": 'red'
}

# Example specific style function for River Segment based on SEGMENT_TYPE
def river_segment_style_function(feature):
    type_color_mapping = {
        'Y': 'blue',   # Real segment
        'N': 'green',  # Real underground segment
        'L': 'purple', # Virtual lake segment
        'T': 'black',  # Virtual transitional water segment
        'C': 'purple', # Virtual coastal water segment
        'V': 'gray',   # Unclassified virtual segment
        None: 'gray'   # Default color for features with None SEGMENT_TYPE
    }
    type_value = feature['properties'].get('SEGMENT_TYPE', None)  # None if type not specified
    
    

    return {
        'fillColor': type_color_mapping.get(type_value, 'gray'),
        'color': type_color_mapping.get(type_value, 'gray'),
        'weight': 1,
        'fillOpacity': 0.5
    }

# %%
import folium
from folium import LayerControl
from IPython.display import display, HTML

# Create a base map
# Create a base map
m = folium.Map(location=[51.5, -0.12], zoom_start=10)

# Add each GeoDataFrame as a separate layer
for name, gdf in gdfs.items():
    if name == "River Segment":
        # Add river segments layer with different SEGMENT_TYPE values if available
        for segment_type in gdf["SEGMENT_TYPE"].dropna().unique():  # Drop null values
            segment_gdf = gdf[gdf["SEGMENT_TYPE"] == segment_type]
            folium.GeoJson(
                segment_gdf,
                name=f"River Segment - {segment_type}",
                style_function=river_segment_style_function,
                tooltip=folium.GeoJsonTooltip(fields=['SEGMENT_TYPE'])
            ).add_to(m)
    else:
        color = layer_colors.get(name, 'blue')  # Default to blue if color not specified
        folium.GeoJson(
            gdf,
            name=name,
            style_function=generic_style_function(color)
        ).add_to(m)

# Add layer control to the map
folium.LayerControl().add_to(m)

# Display the map
m
# %%
