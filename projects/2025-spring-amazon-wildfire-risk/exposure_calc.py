import concurrent.futures as cf
import logging
import os
import gc
import time
from typing import Dict, List, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import rioxarray # noqa - Required for xr.open_dataset with spatial data
import xarray as xr
import xvec
from shapely.geometry import Point
from shapely.ops import transform

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
# Use an environment variable or direct assignment for the ID column
# Ensure your input CSV has a unique identifier column with this name.
ID_COLUMN = "osm_id" # IMPORTANT: Make sure this matches your CSV header for unique IDs
GEOMETRY_COLUMN = "geometry"
S3_BUCKET = os.environ.get("S3_BUCKET", "your-default-bucket") # Use env var or replace default

# Climate data variables to process
# Separating variables for point extraction vs. radius analysis
POINT_DATA_VARS = ['burn_probability_current', 'burn_probability_future_2030', 'fwi_current', 'fwi_future_2030']
RADIUS_DATA_VARS = ['burn_probability_current', 'burn_probability_future_2030'] # Only these get radius stats

# Define radius distances (in miles) and stats for radius analysis
RADIUS_MILES = [5, 10, 25]
RADIUS_STATS = ['max'] # Calculate both mean and max in one pass

# Coordinate Reference Systems
WGS84_CRS = "EPSG:4326"
# NAD83 / Washington North (ftUS) -> Changed to meters version EPSG:2926 is ftUS, 2285 is meters
# Using meters is generally preferred for geo calculations. Confirmed by user ok for E. Washington.
PROJECTED_CRS = "EPSG:2285" # NAD83 / Washington North (meters)

# Input/Output File Paths (adjust as needed)
FACILITY_CSV_PATH = "data/amazon_facilities_eastern_washington.csv"
FIRESTATIONS_GEOJSON_PATH = "data/fire_stations.geojson"
SUBSTATIONS_CSV_PATH = "data/osm_substations.csv"
OUTPUT_EXCEL_PATH = "data/amazon_facilities_with_detailed_fire_exposure.xlsx"
# --- End Configuration ---

def miles_to_meters(miles: Union[int, float]) -> float:
    """Convert miles to meters"""
    return miles * 1609.34

def convert_ds_to_df(ds: xr.Dataset, id_column: str) -> pd.DataFrame:
    """
    Converts an xarray Dataset (typically from xvec) to a pandas DataFrame,
    resetting the index and keeping relevant columns.
    Assumes the Dataset index name matches the provided id_column.
    """
    if not isinstance(ds, xr.Dataset):
        logger.error("Input to convert_ds_to_df is not an xarray Dataset.")
        return pd.DataFrame()

    try:
        # xvec adds stat prefix (e.g., 'mean_') to data vars
        # Keep all data variables calculated
        stat_columns = list(ds.data_vars.keys())
        
        # Convert to DataFrame, assumes index name is set correctly before calling xvec
        df = ds.to_dataframe()
        
        # Check if index needs resetting (depends on how xvec was called)
        if isinstance(df.index, pd.MultiIndex):
             df = df.reset_index()
        else:
             # If single index, reset and potentially rename 'index' column
             df = df.reset_index()
             if 'index' in df.columns and id_column not in df.columns:
                 df = df.rename(columns={'index': id_column})


        # Define columns to keep: ID, month (if exists), and all calculated stats
        columns_to_keep = [id_column]
        if 'month' in df.columns:
            columns_to_keep.append('month')
        columns_to_keep.extend(stat_columns)

        # Ensure we only select columns that actually exist in the DataFrame
        existing_columns_to_keep = [col for col in columns_to_keep if col in df.columns]
        
        if id_column not in df.columns:
             logger.warning(f"ID column '{id_column}' not found in DataFrame after conversion. Check indexing.")
             # Return empty or partial dataframe? Returning empty to signal issue clearly.
             return pd.DataFrame(columns=existing_columns_to_keep)


        return df[existing_columns_to_keep]

    except Exception as e:
        logger.exception(f"Error converting Dataset to DataFrame: {e}")
        return pd.DataFrame()


def task_xvec_zonal_stats(
    climate_subset: xr.Dataset, # Pass only the necessary variables
    geometry_chunk: gpd.GeoSeries,
    x_dim: str,
    y_dim: str,
    stats_list: List[str], # Expecting ['mean', 'max'] etc.
    index_name: str,
    method: str = "exactextract",
) -> pd.DataFrame:
    """
    Worker task for running xvec.zonal_stats in parallel.
    Takes a subset of the climate data.
    """
    try:
        # Ensure geometry chunk has the correct index name for xvec
        geometry_chunk.index.name = index_name

        ds = climate_subset.xvec.zonal_stats(
            geometry_chunk,
            x_coords=x_dim,
            y_coords=y_dim,
            stats=stats_list, # Pass the list of stats
            method=method,
            index=True, # Use the geometry index
        )
        df = convert_ds_to_df(ds=ds, id_column=index_name)
        return df
    except Exception as e:
        logger.error(f"Error in task_xvec_zonal_stats: {e}", exc_info=True)
        # Return an empty dataframe or re-raise depending on desired error handling
        return pd.DataFrame()


def zonal_aggregation_point(
    climate: xr.Dataset,
    infra: gpd.GeoDataFrame,
    x_dim: str,
    y_dim: str,
    data_vars: List[str],
    id_column: str,
) -> pd.DataFrame:
    """Extract values at point locations for specified data variables."""
    logger.info(f"Extracting point values for {len(infra)} facilities for variables: {data_vars}")
    if infra.empty:
        logger.warning("Input GeoDataFrame for point extraction is empty.")
        return pd.DataFrame()
    if not data_vars:
         logger.warning("No data variables specified for point extraction.")
         return pd.DataFrame()

    # Select only the required variables
    climate_subset = climate[data_vars]

    try:
        # Ensure the GeoDataFrame index is set correctly
        if infra.index.name != id_column:
             logger.warning(f"Input GDF index name is '{infra.index.name}', expected '{id_column}'. Setting index.")
             infra = infra.set_index(id_column) # Keep column if needed elsewhere

        ds = climate_subset.xvec.extract_points(
            infra.geometry, x_coords=x_dim, y_coords=y_dim, index=True
        )
        df = convert_ds_to_df(ds=ds, id_column=id_column)
        logger.info(f"Successfully extracted point values for {len(df)} records.")
        return df
    except Exception as e:
        logger.exception(f"Error during point extraction: {e}")
        return pd.DataFrame()

def zonal_aggregation_polygon(
    climate: xr.Dataset,
    infra: gpd.GeoDataFrame, # Expecting buffered polygons
    x_dim: str,
    y_dim: str,
    zonal_agg_methods: List[str], # Expecting ['mean', 'max']
    data_vars: List[str], # Specify variables for zonal stats
    id_column: str,
) -> pd.DataFrame:
    """Perform zonal aggregation on polygon geometries using parallel processing."""
    logger.info(f"Starting zonal aggregation ({', '.join(zonal_agg_methods)}) for {len(infra)} polygons for variables: {data_vars}")
    if infra.empty:
        logger.warning("Input GeoDataFrame for polygon aggregation is empty.")
        return pd.DataFrame()
    if not data_vars:
        logger.warning("No data variables specified for polygon aggregation.")
        return pd.DataFrame()
    
    # Set index correctly before splitting
    if infra.index.name != id_column:
         logger.warning(f"Input GDF index name is '{infra.index.name}', expected '{id_column}'. Setting index.")
         infra = infra.set_index(id_column, drop=False) # Keep column if needed elsewhere

    ds = climate.xvec.zonal_stats(
            infra.geometry,
            x_coords=x_dim,
            y_coords=y_dim,
            stats=RADIUS_STATS, # Pass the list of stats
            method='exactextract',
            index=True, # Use the geometry index
        )
    df_polygon = convert_ds_to_df(ds=ds, id_column=id_column)
    # If index was lost somehow during concat, try resetting
    if id_column not in df_polygon.columns and id_column != df_polygon.index.name:
         df_polygon = df_polygon.reset_index()
         if 'index' in df_polygon.columns:
              df_polygon = df_polygon.rename(columns={'index': id_column})

    logger.info(f"Successfully aggregated polygon values for {len(df_polygon)} records.")
    return df_polygon

def create_buffer_gdfs_vectorized(infra_gdf: gpd.GeoDataFrame, radius_miles_list: List[int], id_column: str) -> Dict[int, gpd.GeoDataFrame]:
    """
    Creates GeoDataFrames with buffered polygons for each radius using vectorized operations.
    Assumes infra_gdf is in WGS84_CRS.
    """
    logger.info(f"Creating vectorized buffers for radii: {radius_miles_list} miles.")
    buffer_gdfs = {}
    original_crs = infra_gdf.crs

    if not original_crs:
         logger.warning("Input GDF CRS is not set. Assuming WGS84 (EPSG:4326).")
         infra_gdf.crs = WGS84_CRS # Set CRS if missing
         original_crs = WGS84_CRS
    elif str(original_crs).upper() != WGS84_CRS:
         logger.warning(f"Input GDF CRS is {original_crs}. Re-projecting to {WGS84_CRS} before buffering.")
         infra_gdf = infra_gdf.to_crs(WGS84_CRS)
         original_crs = WGS84_CRS


    try:
        # Project ONCE to the projected CRS for buffering
        infra_proj = infra_gdf.to_crs(PROJECTED_CRS)
        logger.info(f"Projected {len(infra_proj)} facilities to {PROJECTED_CRS} for buffering.")

        for radius in radius_miles_list:
            start_time = time.time()
            radius_m = miles_to_meters(radius)

            # Create a copy to store the buffered geometry for this radius
            buffer_gdf_proj = infra_proj[[id_column, GEOMETRY_COLUMN]].copy()

            # Buffer ONCE (vectorized, in meters)
            buffer_gdf_proj[GEOMETRY_COLUMN] = infra_proj.geometry.buffer(radius_m)

            # Project back ONCE to the original CRS (WGS84)
            buffer_gdf = buffer_gdf_proj.to_crs(original_crs)

            # Add radius identifier
            buffer_gdf[f'radius_miles'] = radius
            buffer_gdfs[radius] = buffer_gdf
            logger.info(f"Created {radius}-mile buffer GDF in {time.time() - start_time:.2f} seconds.")

    except Exception as e:
        logger.exception(f"Error during vectorized buffer creation: {e}")
        # Depending on requirements, could return partial results or raise error
        return {} # Return empty dict on failure

    return buffer_gdfs

def calculate_substation_metrics(
    infra_gdf: gpd.GeoDataFrame,
    substations_gdf: gpd.GeoDataFrame,
    buffer_gdfs: Dict[int, gpd.GeoDataFrame],
    id_column: str
) -> Tuple[pd.DataFrame, Dict[int, pd.DataFrame]]:
    """
    Calculate electrical infrastructure metrics:
    1. Distance to nearest substation (in miles)
    2. Count of substations within each radius
    
    Args:
        infra_gdf: GeoDataFrame of facility points
        substations_gdf: GeoDataFrame of electrical substations
        buffer_gdfs: Dictionary of buffer GeoDataFrames by radius
        id_column: Name of the unique ID column
        
    Returns:
        Tuple containing:
        - DataFrame with nearest substation distances
        - Dictionary of DataFrames with substation counts by radius
    """
    logger.info(f"Calculating electrical infrastructure metrics for {len(infra_gdf)} facilities.")
    start_time = time.time()
    
    # Initialize results
    distance_df = pd.DataFrame()
    counts_dict = {}
    
    if infra_gdf.empty or substations_gdf.empty:
        logger.warning("Input infra_gdf or substations_gdf is empty. Cannot calculate metrics.")
        # Return empty results
        return (
            pd.DataFrame({id_column: infra_gdf[id_column], 'Distance_to_Substation_miles': np.nan}),
            {radius: pd.DataFrame({id_column: infra_gdf[id_column], f'Substation_Count_{radius}mi': 0, 
                                 f'Substation_Redundant_{radius}mi': False}) for radius in buffer_gdfs.keys()}
        )
    
    # Ensure CRS match
    if infra_gdf.crs is None:
        logger.warning("infra_gdf CRS not set, assuming WGS84.")
        infra_gdf.crs = WGS84_CRS
    if substations_gdf.crs is None:
        logger.warning("substations_gdf CRS not set, assuming WGS84.")
        substations_gdf.crs = WGS84_CRS
    
    if infra_gdf.crs != substations_gdf.crs:
        logger.info(f"Projecting substations from {substations_gdf.crs} to {infra_gdf.crs}.")
        substations_gdf = substations_gdf.to_crs(infra_gdf.crs)
    
    try:
        # 1. Calculate distance to nearest substation (similar to fire station function)
        # Project both to projected CRS for accurate distance calculation
        infra_proj = infra_gdf.to_crs(PROJECTED_CRS)
        substations_proj = substations_gdf.to_crs(PROJECTED_CRS)
        
        # Use spatial join with nearest neighbor
        nearest_join = gpd.sjoin_nearest(
            infra_proj,
            substations_proj,
            how='left',
            distance_col="distance_meters"
        )
        
        # Convert distance from meters to miles
        nearest_join['Distance_to_Substation_miles'] = nearest_join['distance_meters'] * 0.000621371
        
        # Handle null/infinite values
        nearest_join['Distance_to_Substation_miles'].replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Ensure one result per facility
        nearest_join = nearest_join[~nearest_join.index.duplicated(keep='first')]
        
        # Select and return necessary columns
        distance_df = infra_gdf[[id_column]].merge(
            nearest_join[['Distance_to_Substation_miles']],
            left_index=True,
            right_index=True,
            how='left'
        )
        
        # 2. Count substations within each buffer radius
        for radius, buffer_gdf in buffer_gdfs.items():
            # Ensure buffer GDF is in the same CRS as substations
            if buffer_gdf.crs != substations_gdf.crs:
                buffer_gdf = buffer_gdf.to_crs(substations_gdf.crs)
            
            # Spatial join to find substations within each buffer
            joined = gpd.sjoin(buffer_gdf, substations_gdf, how='left', predicate='contains')
            
            # Count substations per facility
            counts = joined.groupby(id_column).size().reset_index(name=f'Substation_Count_{radius}mileRadius')
            
            # Handle facilities with no substations in radius (will be missing from join result)
            counts = infra_gdf[[id_column]].merge(counts, on=id_column, how='left')
            
            # Fill NaN with zeros (no substations in radius)
            counts[f'Substation_Count_{radius}mileRadius'] = counts[f'Substation_Count_{radius}mileRadius'].fillna(0).astype(int)
            
            counts_dict[radius] = counts
        
        logger.info(f"Calculated electrical infrastructure metrics in {time.time() - start_time:.2f} seconds.")
        
    except Exception as e:
        logger.exception(f"Error calculating electrical infrastructure metrics: {e}")
        # Return empty DataFrames
        distance_df = pd.DataFrame({id_column: infra_gdf[id_column], 'Distance_to_Substation_miles': np.nan})
        counts_dict = {radius: pd.DataFrame({id_column: infra_gdf[id_column], 
                                           f'Substation_Count_{radius}mileRadius': 0,}) 
                      for radius in buffer_gdfs.keys()}
    
    return distance_df, counts_dict

def calculate_distance_to_nearest_firestation(
    infra_gdf: gpd.GeoDataFrame,
    firestations_gdf: gpd.GeoDataFrame,
    id_column: str
) -> pd.DataFrame:
    """
    Calculate the distance from each facility to the nearest fire station in miles
    using efficient spatial join and accurate projection.
    """
    logger.info(f"Calculating distance to nearest fire station for {len(infra_gdf)} facilities.")
    start_time = time.time()

    if infra_gdf.empty or firestations_gdf.empty:
        logger.warning("Input infra_gdf or firestations_gdf is empty. Cannot calculate distances.")
        # Return DataFrame with NaNs matching infra_gdf index
        return pd.DataFrame({id_column: infra_gdf[id_column], 'Distance_to_FireStation_miles': np.nan}).set_index(id_column)

    # Ensure inputs are GeoDataFrames
    if not isinstance(infra_gdf, gpd.GeoDataFrame) or not isinstance(firestations_gdf, gpd.GeoDataFrame):
        raise TypeError("Both infra_gdf and firestations_gdf must be GeoDataFrames.")

    # Ensure CRS are set and match, or project firestations to infra CRS if needed
    if infra_gdf.crs is None:
         logger.warning("infra_gdf CRS not set, assuming WGS84.")
         infra_gdf.crs = WGS84_CRS
    if firestations_gdf.crs is None:
         logger.warning("firestations_gdf CRS not set, assuming WGS84.")
         firestations_gdf.crs = WGS84_CRS

    if infra_gdf.crs != firestations_gdf.crs:
        logger.info(f"Projecting fire stations from {firestations_gdf.crs} to {infra_gdf.crs}.")
        firestations_gdf = firestations_gdf.to_crs(infra_gdf.crs)

    try:
        # Project both to the chosen projected CRS for accurate distance calculation
        infra_proj = infra_gdf.to_crs(PROJECTED_CRS)
        firestations_proj = firestations_gdf.to_crs(PROJECTED_CRS)
        logger.info(f"Projected facilities and fire stations to {PROJECTED_CRS} for distance calculation.")

        # Perform the nearest spatial join
        # `sjoin_nearest` keeps the left GDF's geometry and index
        # distance_col reports distance in the units of the GDF's CRS (meters in this case)
        joined_gdf = gpd.sjoin_nearest(
            infra_proj,
            firestations_proj,
            how='left',
            distance_col="distance_meters"
        )
        logger.info("Performed nearest join between facilities and fire stations.")

        # Convert distance from meters to miles
        # 1 meter = 0.000621371 miles
        joined_gdf['Distance_to_FireStation_miles'] = joined_gdf['distance_meters'] * 0.000621371

        # Handle cases where no fire station is found (sjoin_nearest might produce NaNs or Inf)
        # Check for NaN distance_meters before conversion if necessary
        joined_gdf['Distance_to_FireStation_miles'].replace([np.inf, -np.inf], np.nan, inplace=True)

        # Ensure one result per original facility: sjoin_nearest can duplicate left points
        # if multiple right points are equidistant. Keep the first match.
        # We sort by distance first to ensure we keep the *closest* if there are multiple joins
        # within tolerance (though sjoin_nearest usually handles this)
        # joined_gdf = joined_gdf.sort_values(by="distance_meters") # Optional sorting
        
        # Crucially, use the original index of infra_gdf to deduplicate
        joined_gdf = joined_gdf[~joined_gdf.index.duplicated(keep='first')]

        # Select and return only the necessary columns, using the original infra index
        # Merge back with original infra GDF's ID column to ensure it's correct
        result_df = infra_gdf[[id_column]].merge(
             joined_gdf[['Distance_to_FireStation_miles']],
             left_index=True,
             right_index=True,
             how='left'
        )
        
        logger.info(f"Calculated distances in {time.time() - start_time:.2f} seconds.")
        return result_df

    except Exception as e:
        logger.exception(f"Error calculating distance to nearest fire station: {e}")
        # Return DataFrame with NaNs matching infra_gdf index
        return pd.DataFrame({id_column: infra_gdf[id_column], 'Distance_to_FireStation_miles': np.nan}).set_index(id_column)

def main():
    """Main function to process and analyze wildfire risk data"""
    start_time_main = time.time()
    logger.info("Starting wildfire risk analysis script.")

    # --- 1. Load Facility Data ---
    try:
        infra_df = pd.read_csv(FACILITY_CSV_PATH)
        logger.info(f"Loaded facility data from {FACILITY_CSV_PATH}")
        
        # Validate ID_COLUMN existence
        if ID_COLUMN not in infra_df.columns:
             logger.error(f"ID Column '{ID_COLUMN}' not found in {FACILITY_CSV_PATH}. Please check column names.")
             return # Stop execution
        
        # Ensure ID column is suitable as index (unique, non-null)
        if not infra_df[ID_COLUMN].is_unique:
             logger.warning(f"Values in ID column '{ID_COLUMN}' are not unique. Merging might produce unexpected results.")
        if infra_df[ID_COLUMN].isnull().any():
             logger.error(f"ID column '{ID_COLUMN}' contains null values. Cannot proceed.")
             return

        infra_gdf = gpd.GeoDataFrame(
            infra_df,
            geometry=gpd.points_from_xy(x=infra_df["longitude"], y=infra_df["latitude"]),
            crs=WGS84_CRS # Assume input lat/lon is WGS84
        )
        # Set the index to the specified ID column *early*
        #infra_gdf = infra_gdf.set_index(ID_COLUMN, drop=False) # Keep column for merging later
        logger.info(f"Created GeoDataFrame for {len(infra_gdf)} facilities with index '{ID_COLUMN}'.")

    except FileNotFoundError:
        logger.error(f"Facility data file not found: {FACILITY_CSV_PATH}")
        return
    except Exception as e:
        logger.exception(f"Error loading or processing facility data: {e}")
        return

    # --- 2. Load Climate Data ---
    try:
        # Construct S3 path if needed
        s3_path = f"s3://{S3_BUCKET}/student-projects/amazon-wildfire-risk-spring2025/data/cmip6_adjusted_burn_probability.zarr"
        logger.info(f"Attempting to load climate data from: {s3_path}")
        climate_ds = xr.open_dataset(s3_path, engine="zarr") # Chunking by month can help memory
        
        # Ensure CRS is set if not automatically detected (rioxarray helps here)
        if climate_ds.rio.crs is None:
            logger.warning("Climate data CRS not found, attempting to set WGS84")
            # This assumes data is WGS84 if not specified; adjust if known otherwise
            climate_ds = climate_ds.rio.write_crs(WGS84_CRS, inplace=True) 
        
        # Rename spatial dims if needed to match xvec defaults ('x', 'y')
        # Check current names and rename if necessary (example)
        # if 'longitude' in climate_ds.dims: climate_ds = climate_ds.rename({'longitude': 'x'})
        # if 'latitude' in climate_ds.dims: climate_ds = climate_ds.rename({'latitude': 'y'})
        # Assuming dims are already 'x', 'y' based on original script context
        X_DIM, Y_DIM = "x", "y" 
        
        logger.info(f"Climate data loaded successfully. Variables: {list(climate_ds.data_vars)}")
        logger.info(f"Climate data CRS: {climate_ds.rio.crs}, Spatial dimensions: {X_DIM}, {Y_DIM}")

    except Exception as e:
        logger.exception(f"Error loading climate data from S3: {e}. Check S3 path, credentials, and dependencies.")
        return

    # --- 3. Load Fire Station Data & Calculate Distances ---
    firestation_distance_df = pd.DataFrame() # Initialize empty
    try:
        logger.info(f"Loading fire station data from: {FIRESTATIONS_GEOJSON_PATH}")
        firestations_gdf = gpd.read_file(FIRESTATIONS_GEOJSON_PATH)
        logger.info(f"Loaded {len(firestations_gdf)} fire station locations.")

        # Calculate distance to nearest fire station (uses infra_gdf index)
        firestation_distance_df = calculate_distance_to_nearest_firestation(
            infra_gdf, # Pass original GDF with correct index
            firestations_gdf,
            id_column=ID_COLUMN
        )
        # Result has ID_COLUMN and Distance_to_FireStation_miles

    except FileNotFoundError:
        logger.warning(f"Fire station data file not found: {FIRESTATIONS_GEOJSON_PATH}. Distance will be NaN.")
    except Exception as e:
        logger.warning(f"Error loading or processing fire station data: {str(e)}. Distance will be NaN.")
    
    # Create placeholder if distance calculation failed or file not found
    if firestation_distance_df.empty:
         logger.warning("Creating placeholder for Distance_to_FireStation_miles")
         firestation_distance_df = pd.DataFrame({
            ID_COLUMN: infra_gdf.index, # Use the index from infra_gdf
            'Distance_to_FireStation_miles': np.nan
         }).set_index(ID_COLUMN)


    # --- 4. Extract Point Values (FWI, Burn Prob at Location) ---
    logger.info("Starting point value extraction for facility locations...")
    point_df = zonal_aggregation_point(
        climate=climate_ds,
        infra=infra_gdf, # Pass GDF with correct index
        x_dim=X_DIM,
        y_dim=Y_DIM,
        data_vars=POINT_DATA_VARS, # Extract all point vars
        id_column=ID_COLUMN
    )
    point_df = point_df.fillna(0)
    # Rename point value columns for clarity
    point_rename_map = {var: f"{var}_point" for var in POINT_DATA_VARS if var in point_df.columns}
    point_df = point_df.rename(columns=point_rename_map)
    logger.info(f"Finished point extraction. Result columns: {point_df.columns.tolist()}")


    # --- 5. Create Buffers ---
    # Use the more efficient vectorized approach
    buffer_gdfs = create_buffer_gdfs_vectorized(infra_gdf, RADIUS_MILES, id_column=ID_COLUMN)
    # buffer_gdfs is a dict: {5: GeoDataFrame, 10: GeoDataFrame, 25: GeoDataFrame}

    # --- 6. Load Substation Data & Calculate Metrics ---
    try:
        logger.info("Loading electrical substation data...")
        substations_df = pd.read_csv(SUBSTATIONS_CSV_PATH)
        substations_gdf = gpd.GeoDataFrame(data=substations_df, geometry=gpd.points_from_xy(x=substations_df['longitude'], y=substations_df['latitude']))
        substations_gdf = substations_gdf.drop([ID_COLUMN, "latitude", "longitude"], axis=1)
        logger.info(f"Loaded {len(substations_gdf)} electrical substations.")
        
        # Calculate substation metrics
        substation_distance_df, substation_counts_dict = calculate_substation_metrics(
            infra_gdf,
            substations_gdf,
            buffer_gdfs,
            id_column=ID_COLUMN
        )
        
    except FileNotFoundError:
        logger.warning("Electrical substation data file not found. Substation metrics will be NaN/zeros.")
        # Create empty results
        substation_distance_df = pd.DataFrame({ID_COLUMN: infra_gdf[ID_COLUMN], 'Distance_to_Substation_miles': np.nan})
        substation_counts_dict = {radius: pd.DataFrame({ID_COLUMN: infra_gdf[ID_COLUMN], 
                                                    f'Substation_Count_{radius}mileRadius': 0,}) 
                                for radius in RADIUS_MILES}
    except Exception as e:
        logger.warning(f"Error processing substation data: {str(e)}. Substation metrics will be NaN/zeros.")
        # Create empty results same as above


    # --- 7. Process Each Radius (Zonal Stats for Burn Probability) ---
    # Select only the required variables and compute *once* before parallelizing
    radius_results_dict = {} # Store results DataFrames for each radius
    for radius in RADIUS_MILES:
        radius_start_time = time.time()
        logger.info(f"--- Starting zonal aggregation for {radius}-mile radius ---")

        if radius not in buffer_gdfs or buffer_gdfs[radius].empty:
             logger.warning(f"Buffer GDF for radius {radius} is missing or empty. Skipping.")
             continue # Skip to next radius if buffer creation failed

        buffer_gdf_for_radius = buffer_gdfs[radius]
        # Ensure index is correct before passing to aggregation
        buffer_gdf_for_radius = buffer_gdf_for_radius.set_index(ID_COLUMN, drop=False)
        months_results_list = []
        for month in climate_ds['month'].values:
            climate_subset = climate_ds.sel(month=month)[RADIUS_DATA_VARS].compute()
            # Perform zonal aggregation for MEAN and MAX in one go, only for RADIUS_DATA_VARS
            stats_df = zonal_aggregation_polygon(
                climate=climate_subset, # Pass the full dataset, selection happens inside
                infra=buffer_gdf_for_radius,
                x_dim=X_DIM,
                y_dim=Y_DIM,
                zonal_agg_methods=RADIUS_STATS, # ['mean', 'max']
                data_vars=RADIUS_DATA_VARS, # Only 'burn_probability_*'
                id_column=ID_COLUMN
            )

            stats_df = stats_df.fillna(0)
            if stats_df.empty:
                logger.warning(f"Zonal aggregation for radius {radius} returned no results. Skipping merge for this radius.")
                continue
            
            # Select only the necessary columns for merging (ID, month, and the renamed stats)
            columns_to_keep = [ID_COLUMN, "month"]

            # --- RENAME COLUMNS HERE --- (Moved renaming outside inner loop)
            rename_map = {}
            # Input cols will be like 'max_burn_probability_current'
            # Output cols should be like 'Max_Burn_Prob_Current_Xmi'
            for var in RADIUS_DATA_VARS:
                original_col = f"{var}" # Specific to RADIUS_STATS = ['max']
                time_period = "Future_2030" if "future_2030" in var else "Current"
                new_col = f"Burn_Probability_{time_period}_{radius}mileRadius"
                if original_col in stats_df.columns:
                    rename_map[original_col] = new_col
                    columns_to_keep.append(new_col)
                else:
                    logger.warning(f"Expected column '{original_col}' not found after monthly concat for radius {radius}.")
            
            stats_df = stats_df.rename(columns=rename_map)
            stats_df["month"] = month

            # Ensure columns exist before selection
            existing_columns_to_keep = [col for col in columns_to_keep if col in stats_df.columns]
            months_results_list.append(stats_df[existing_columns_to_keep])
            del climate_subset
            gc.collect()
        months_df = pd.concat(months_results_list)
        radius_results_dict[radius] = months_df
        logger.info(f"--- Finished processing {radius}-mile radius in {time.time() - radius_start_time:.2f} seconds ---")

    # --- 8. Combine All Results ---
    logger.info("Combining all results...")

    # Start with the original facility attributes (non-spatial)
    # Keep only necessary original columns + ID
    # final_df = infra_df[[ID_COLUMN, 'original_attribute1', 'original_attribute2']].copy()
    final_df = infra_df.copy() # Or select specific columns if needed
    logger.info(f"Starting merge with {len(final_df)} original facility records.")
    
    # Ensure ID_COLUMN is the index for merging distance data
    if final_df.index.name != ID_COLUMN:
        final_df = final_df.set_index(ID_COLUMN)
        
    # Add fire station distances (merge on index ID_COLUMN)
    if not firestation_distance_df.empty:
        # Ensure distance df is indexed correctly
        if firestation_distance_df.index.name != ID_COLUMN:
             firestation_distance_df = firestation_distance_df.set_index(ID_COLUMN)
        final_df = final_df.merge(firestation_distance_df[['Distance_to_FireStation_miles']],
                                  left_index=True, right_index=True, how='left')
        logger.info(f"Merged fire station distances. DataFrame shape: {final_df.shape}")
    else:
        logger.warning("Skipping merge of fire station distances (data was empty).")
        final_df['Distance_to_FireStation_miles'] = np.nan # Add column explicitly if empty

    # Add substation distance metrics
    if not substation_distance_df.empty:
        if substation_distance_df.index.name != ID_COLUMN:
            substation_distance_df = substation_distance_df.set_index(ID_COLUMN)
        final_df = final_df.merge(substation_distance_df[['Distance_to_Substation_miles']],
                                left_index=True, right_index=True, how='left')
        logger.info(f"Merged substation distances. DataFrame shape: {final_df.shape}")

    # Merge requires resetting index if merging on columns including month
    final_df = final_df.reset_index()

    # Add substation count metrics for each radius
    for radius, count_df in substation_counts_dict.items():
        if not count_df.empty:
            merge_cols = [ID_COLUMN]
            if all(col in final_df.columns for col in merge_cols):
                final_df = pd.merge(final_df, count_df, on=merge_cols, how='left')
                logger.info(f"Merged substation count data for {radius}-mile radius.")

    

    # Add point values (merge on ID_COLUMN and month)
    if not point_df.empty:
        merge_cols_point = [ID_COLUMN]
        
        if all(col in final_df.columns for col in merge_cols_point):
             final_df = pd.merge(final_df, point_df, on=merge_cols_point, how='left')
             logger.info(f"Merged point data. DataFrame shape: {final_df.shape}")
        else:
             logger.warning(f"Cannot merge point data. Missing one or more merge columns ({merge_cols_point}) in final_df.")
             # Add NaN columns for point data if merge fails?
             for col in point_rename_map.values(): final_df[col] = np.nan

    else:
        logger.warning("Skipping merge of point data (data was empty).")
        # Add NaN columns if data was expected
        for col in point_rename_map.values(): final_df[col] = np.nan

    # Add radius-based values (merge on ID_COLUMN and month)
    for radius, radius_df in radius_results_dict.items():
        if not radius_df.empty:
            merge_cols_radius = [ID_COLUMN]
            if 'month' in radius_df.columns:
                 merge_cols_radius.append('month')
            
            if all(col in final_df.columns for col in merge_cols_radius):    
                final_df = pd.merge(final_df, radius_df, on=merge_cols_radius, how='left')
                logger.info(f"Merged radius {radius} data. DataFrame shape: {final_df.shape}")
            else:
                 logger.warning(f"Cannot merge radius {radius} data. Missing one or more merge columns ({merge_cols_radius}) in final_df.")
                 # Add NaN columns for this radius data if merge fails?

        else:
            logger.warning(f"Skipping merge for radius {radius} (data was empty).")
            # Optionally add NaN columns for this radius

    # --- 9. Save the Results ---
    try:
        # Check for potential duplicate columns before saving (can happen if merges go wrong)
        final_columns = final_df.columns
        if len(final_columns) != len(set(final_columns)):
             logger.warning(f"Duplicate columns found in final DataFrame: {[col for col in final_columns if list(final_columns).count(col) > 1]}")
             # Could implement logic to drop duplicates here if needed
             # final_df = final_df.loc[:,~final_df.columns.duplicated()]


        logger.info(f"Saving final results ({final_df.shape[0]} rows, {final_df.shape[1]} columns) to Excel: {OUTPUT_EXCEL_PATH}")
        # Ensure directory exists
        os.makedirs(os.path.dirname(OUTPUT_EXCEL_PATH), exist_ok=True)
        
        # Save to Excel format as requested
        final_df.to_excel(OUTPUT_EXCEL_PATH, index=False, engine='openpyxl') # Requires openpyxl
        
        total_time = time.time() - start_time_main
        logger.info(f"Analysis complete. Data saved successfully in {total_time:.2f} seconds.")

    except ImportError:
         logger.error("`openpyxl` library not found. Cannot save to Excel. Please install it: pip install openpyxl")
         logger.info("Attempting to save as CSV instead.")
         csv_fallback_path = OUTPUT_EXCEL_PATH.replace(".xlsx", ".csv")
         try:
             final_df.to_csv(csv_fallback_path, index=False)
             logger.info(f"Saved results as CSV: {csv_fallback_path}")
         except Exception as e_csv:
             logger.exception(f"Failed to save as CSV as well: {e_csv}")
    except Exception as e:
        logger.exception(f"Error saving results to Excel: {e}")


if __name__ == "__main__":
    # Ensure necessary directories exist
    os.makedirs("data", exist_ok=True)
    
    # Run the main analysis
    main()