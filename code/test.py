# %%

import geopandas as gpd

# Define the paths to the files
catchment_path = "C:/Users/Dpayn/MLE/Machine-Learning/Large_data/CatchmentStakeholderGroup.gml"
water_supplies_path = "C:/Users/Dpayn/MLE/Machine-Learning/Large_data/Register_of_Private_Water_Supplies_in_Northern_Ireland_(24_04_2020).geojson"
river_segment_path = "C:/Users/Dpayn/MLE/Machine-Learning/Large_data/RiverSegment.gml"
transitional_water_bodies_path = "C:/Users/Dpayn/MLE/Machine-Learning/Large_data/TransitionalWaterBodies-1st-cycle.gml"

# Load the GeoDataFrames
catchment_gdf = gpd.read_file(catchment_path)
water_supplies_gdf = gpd.read_file(water_supplies_path)
river_segment_gdf = gpd.read_file(river_segment_path)
transitional_water_bodies_gdf = gpd.read_file(transitional_water_bodies_path)

# Ensure all GeoDataFrames use the same CRS
if catchment_gdf.crs != water_supplies_gdf.crs:
    water_supplies_gdf = water_supplies_gdf.to_crs(catchment_gdf.crs)

if catchment_gdf.crs != river_segment_gdf.crs:
    river_segment_gdf = river_segment_gdf.to_crs(catchment_gdf.crs)

if catchment_gdf.crs != transitional_water_bodies_gdf.crs:
    transitional_water_bodies_gdf = transitional_water_bodies_gdf.to_crs(catchment_gdf.crs)

# Perform spatial join for water supplies
joined_water_supplies = gpd.sjoin(water_supplies_gdf, catchment_gdf, how="inner", op='within')
count_per_zone_water = joined_water_supplies.groupby('index_right').size()
catchment_gdf['water_supply_count'] = catchment_gdf.index.map(count_per_zone_water).fillna(0)

# Perform spatial join for river segments
joined_river_segments = gpd.sjoin(river_segment_gdf, catchment_gdf, how="inner", op='within')

# Calculate the number of river segments per catchment zone
count_per_zone_river = joined_river_segments.groupby('index_right').size()
catchment_gdf['river_segment_count'] = catchment_gdf.index.map(count_per_zone_river).fillna(0)

# Calculate the total length of river segments per catchment zone
joined_river_segments['length'] = joined_river_segments.geometry.length
length_per_zone_river = joined_river_segments.groupby('index_right')['length'].sum()
catchment_gdf['total_river_length'] = catchment_gdf.index.map(length_per_zone_river).fillna(0)

# Perform spatial join for transitional water bodies
joined_transitional_water_bodies = gpd.sjoin(transitional_water_bodies_gdf, catchment_gdf, how="inner", op='within')

# Calculate the total area of transitional water bodies per catchment zone
joined_transitional_water_bodies['area'] = joined_transitional_water_bodies.geometry.area
area_per_zone_transitional = joined_transitional_water_bodies.groupby('index_right')['area'].sum()
catchment_gdf['total_transitional_water_area'] = catchment_gdf.index.map(area_per_zone_transitional).fillna(0)

# Define the path to save the CSV file
file_path = "C:/Users/Dpayn/MLE/Machine-Learning/Large_data/area_stats.csv"

# Save the results to a CSV file
catchment_gdf[['NAME', 'water_supply_count', 'river_segment_count', 'total_river_length', 'total_transitional_water_area']].to_csv(file_path, index=False)

# %%
