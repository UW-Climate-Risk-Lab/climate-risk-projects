import geopandas as gpd
import pandas as pd
import xarray as xr
import rioxarray
import xvec
import os

from dask.distributed import Client, LocalCluster

from utils import download_and_read_fwi_data_for_dates

# --- Configuration ---
CONFIG = {
    "MTBS_RASTER_DATA_DIR": "./data/mtbs_rasters",
    "MTBS_PERIMETER_DATA_PATH": "./data/mtbs_perimeters/mtbs_perims_DD.shp",
    "EIA_UTILITY_SERVICE_AREAS_PATH": "./data/eia_service_areas/Electric_Retail_Service_Territories.geojson",
    "FWI_VARIABLE": "GPM.LATE.v5_FWI", # Example FWI variable name in the NetCDF
    "MEMORY_PER_WORKER": "30GB",
    "N_WORKERS": 4, # Adjusted for typical local testing; original was 6
    "THREADS_PER_WORKER": 2,
    "CRS": "EPSG:4269",  # Target CRS for most operations
    "FWI_CRS": "EPSG:4326", # Initial CRS of FWI data (WGS84)
    "CACHE_DIR_BASE": "./data",
    "CACHE_SUBDIR_NAME": "iteration_cache",
    "OUTPUT_CSV_PATH": "utility_service_area_fire_metrics.csv.gz",
    "ZONAL_STATS_METHOD": "exactextract", # 'exactextract' or 'rasterstats'
    "FWI_X_SLICE": slice(-125.4, -112.5), # Longitude slice for FWI data (In FWI_CRS degrees)
    "FWI_Y_SLICE": slice(32.4, 50.1),   # Latitude slice for FWI data (In FWI_CRS degrees)
}
CONFIG["CACHE_DIR"] = os.path.join(CONFIG["CACHE_DIR_BASE"], CONFIG["CACHE_SUBDIR_NAME"])

# --- Helper Functions ---


def calculate_zonal_stats_for_df(data_array, gdf_indexed_geom, variable_name_prefix):
    """
    Calculates zonal statistics (mean, max) for a DataArray against indexed geometries.
    Args:
        data_array (xr.DataArray): DataArray with geospatial data (and optionally 'time' coord).
                                   Must have 'x' and 'y' coordinates.
        gdf_indexed_geom (gpd.GeoDataFrame): GeoDataFrame with geometry, indexed by 'Event_ID'.
        variable_name_prefix (str): Prefix for output columns (e.g., "fire_severity", "fwi").
    Returns:
        pd.DataFrame: DataFrame with Event_ID, [time], and mean/max statistics.
    """
    # Ensure data_array has a name for reliable processing in to_dataframe()
    internal_data_array_name = "stat_value"
    data_array = data_array.rename(internal_data_array_name)

    stats_da = data_array.xvec.zonal_stats(
        gdf_indexed_geom.geometry,
        x_coords="x",
        y_coords="y",
        stats=["mean", "max"],
        method=CONFIG["ZONAL_STATS_METHOD"],
        index=True, # Uses index from gdf_indexed_geom (e.g., Event_ID)
    )
    # stats_da now has dimensions (index_name_from_gdf, [time_if_present_in_data_array], zonal_statistics)
    # The values are the statistics. Its name will be internal_data_array_name.
    stats_da.name = variable_name_prefix
    df_temp = stats_da.to_dataframe().reset_index()

    df_mean = df_temp.loc[df_temp["zonal_statistics"]=="mean"]
    df_mean = df_mean.rename(columns={stats_da.name: stats_da.name+"_mean"})

    df_max = df_temp.loc[df_temp["zonal_statistics"]=="max"]
    df_max = df_max.rename(columns={stats_da.name: stats_da.name+"_max"})

    if ("time" in df_max.columns) and ("time" in df_mean.columns):
        stats_df = pd.merge(df_max, df_mean, how="left", on=["Event_ID", "time"])[["Event_ID", "time", stats_da.name+"_mean", stats_da.name+"_max"]]
    else:
        stats_df = pd.merge(df_max, df_mean, how="left", on=["Event_ID"])[["Event_ID", stats_da.name+"_mean", stats_da.name+"_max"]]
    return stats_df


def process_single_event_data(mtbs_raster_path, full_mtbs_perimeters_gdf, config):
    """
    Processes a single MTBS event: loads raster, filters perimeters,
    downloads/processes FWI, calculates zonal stats for fire severity and FWI.
    Caches results and loads from cache if available.
    """
    filename_base = os.path.splitext(os.path.basename(mtbs_raster_path))[0]
    print(f"Processing {filename_base}...")

    os.makedirs(config["CACHE_DIR"], exist_ok=True)
    fire_severity_cache_file = os.path.join(config["CACHE_DIR"], f"fs_{filename_base}.csv.gz")
    fwi_cache_file = os.path.join(config["CACHE_DIR"], f"fwi_{filename_base}.csv.gz")

    if os.path.exists(fire_severity_cache_file) and os.path.exists(fwi_cache_file):
        print(f"  Loading from cache: {filename_base}")
        df_fire_severity_iter = pd.read_csv(fire_severity_cache_file)
        df_fwi_iter = pd.read_csv(fwi_cache_file)
        if 'time' in df_fwi_iter.columns:
            try:
                df_fwi_iter['time'] = pd.to_datetime(df_fwi_iter['time'])
            except Exception as e:
                print(f"  Warning: Could not parse 'time' column in cached FWI data for {filename_base}: {e}")
        return df_fire_severity_iter, df_fwi_iter

    print(f"  Calculating data for {filename_base} (not found in cache).")
    try:
        # Example filename: ca3986712009520070622_2007_burn_severity.tif -> want state and year
        # Original script: file_info = file[:-4].split("_"), state = file_info[1], year = int(file_info[2])
        # This assumes a fixed format like "PREFIX_STATE_YEAR_ADDITIONALINFO.tif"
        # For a more robust parsing, consider regex or more flexible splitting if format varies.
        # Using a simplified parsing for now, adjust if needed:
        parts = filename_base.split('_')
        state_abbr = parts[1] # Assuming state is at the start of the filename or derivable.
        year_str = parts[2] # Assuming year is the second part.
        year = int(year_str)
    except (IndexError, ValueError) as e:
        print(f"  Could not parse state/year from filename '{filename_base}'. Expecting format like 'CA_2020_...' or similar. Error: {e}. Skipping.")
        return None, None

    # 1. Load and preprocess MTBS raster for fire severity
    da_mtsb_raster = rioxarray.open_rasterio(mtbs_raster_path, masked=True)
    da_mtsb_raster = da_mtsb_raster.rio.reproject(config["CRS"])
    da_mtsb_raster = da_mtsb_raster.where((da_mtsb_raster >= 2) & (da_mtsb_raster <= 4)) # severity 2,3,4
    # Ensure 'x' and 'y' are coordinate names if not already
    da_mtsb_raster = da_mtsb_raster.rename({'x': 'x', 'y': 'y'})


    # 2. Filter MTBS perimeters for the current event context
    gdf_filtered_perimeters = full_mtbs_perimeters_gdf[
        (full_mtbs_perimeters_gdf["Incid_Type"] == "Wildfire") &
        (full_mtbs_perimeters_gdf["Ig_Date"].dt.year == year) &
        (full_mtbs_perimeters_gdf["Event_ID"].str.upper().str.startswith(state_abbr)) # Match Event_ID start with state
    ]

    if gdf_filtered_perimeters.empty:
        print(f"  No matching perimeters found for {state_abbr}, {year} from {filename_base}. Skipping.")
        return None, None

    gdf_event_perimeters = gdf_filtered_perimeters.set_index("Event_ID")[["geometry"]]
    gdf_event_perimeters = gdf_event_perimeters.to_crs(da_mtsb_raster.rio.crs)

    if gdf_event_perimeters.empty: # Double check after reprojection
        print(f"  Perimeters became empty after CRS transformation or initial filtering for {filename_base}. Skipping.")
        return None, None
        
    # 3. Calculate Fire Severity zonal stats
    df_fire_severity_iter = calculate_zonal_stats_for_df(
        da_mtsb_raster, gdf_event_perimeters, "fire_severity" # Squeeze to remove band dim if it exists
    )

    # 4. Download and process FWI data
    ignition_dates = gdf_filtered_perimeters["Ig_Date"].dropna().drop_duplicates().sort_values()
    df_fwi_iter = pd.DataFrame(columns=['Event_ID', 'time', 'fwi_mean', 'fwi_max']) # Default empty
    df_fwi_iter['time'] = pd.to_datetime(df_fwi_iter['time'])


    if ignition_dates.empty:
        print(f"  No valid ignition dates for FWI processing for {filename_base}.")
    else:
        ds_fwi = download_and_read_fwi_data_for_dates(ignition_dates)
        if ds_fwi is None or not ds_fwi.data_vars or config["FWI_VARIABLE"] not in ds_fwi:
            print(f"  FWI data download/read failed or is empty/missing variable for {filename_base}.")
        else:
            ds_fwi = ds_fwi.rio.write_crs(config["FWI_CRS"])
            ds_fwi = ds_fwi.rename({"lat": "y", "lon": "x"}) # Standardize coord names
            da_fwi_raw = ds_fwi[config["FWI_VARIABLE"]]

            if config.get("FWI_X_SLICE") and config.get("FWI_Y_SLICE"):
                # Ensure x, y exist before slicing
                if 'x' in da_fwi_raw.coords and 'y' in da_fwi_raw.coords:
                    da_fwi_raw = da_fwi_raw.sel(x=config["FWI_X_SLICE"], y=config["FWI_Y_SLICE"])
                else:
                    print("  Warning: 'x' or 'y' coordinates not found in FWI data for slicing.")
            
            da_fwi_computed = da_fwi_raw.compute() # Compute if Dask array
            da_fwi_computed = da_fwi_computed.rio.reproject(CONFIG["CRS"])
            bounds = da_mtsb_raster.squeeze().rio.bounds()
            da_fwi_clipped = da_fwi_computed.rio.clip_box(*bounds)
            
            
            # 5. Calculate FWI zonal stats
            if not da_fwi_clipped.x.size == 0 and not da_fwi_clipped.y.size == 0 : # Check if clipped array is not empty
                df_fwi_iter = calculate_zonal_stats_for_df(
                    da_fwi_clipped, gdf_event_perimeters, "fwi"
                )
            else:
                print(f"  FWI data became empty after reprojection/clipping for {filename_base}.")


    # Cache results
    if df_fire_severity_iter is not None and not df_fire_severity_iter.empty:
        df_fire_severity_iter.to_csv(fire_severity_cache_file, index=False, compression="gzip")
    # Save empty FWI df if it's empty, so cache check passes next time for this file
    df_fwi_iter.to_csv(fwi_cache_file, index=False, compression="gzip")
    
    print(f"  Finished processing for {filename_base}. Data cached.")
    del ds_fwi
    del da_mtsb_raster
    return df_fire_severity_iter, df_fwi_iter

# --- Main Script ---
def main():
    print("Starting Dask client...")
    cluster = LocalCluster(
        n_workers=CONFIG["N_WORKERS"],
        threads_per_worker=CONFIG["THREADS_PER_WORKER"],
        memory_limit=CONFIG["MEMORY_PER_WORKER"]
    )
    client = Client(cluster)
    print(f"Dask client started. Dashboard: {client.dashboard_link}")

    print("Loading MTBS perimeter data...")
    mtbs_perimeters_gdf = gpd.read_file(CONFIG["MTBS_PERIMETER_DATA_PATH"])
    mtbs_perimeters_gdf["Ig_Date"] = pd.to_datetime(mtbs_perimeters_gdf["Ig_Date"], errors='coerce')
    mtbs_perimeters_gdf = mtbs_perimeters_gdf.to_crs(CONFIG["CRS"])

    mtbs_raster_files = [
        os.path.join(CONFIG["MTBS_RASTER_DATA_DIR"], f)
        for f in os.listdir(CONFIG["MTBS_RASTER_DATA_DIR"])
        if f.lower().endswith(('.tif', '.tiff', '.img')) # Common raster extensions
    ]
    if not mtbs_raster_files:
        print(f"No raster files found in {CONFIG['MTBS_RASTER_DATA_DIR']}. Exiting.")
        client.close()
        cluster.close()
        return

    all_fire_severity_dfs = []
    all_fwi_dfs = []

    for raster_file_path in mtbs_raster_files:
        df_sev_iter, df_fwi_iter = process_single_event_data(
            raster_file_path, mtbs_perimeters_gdf, CONFIG
        )
        if df_sev_iter is not None and not df_sev_iter.empty:
            all_fire_severity_dfs.append(df_sev_iter)
        if df_fwi_iter is not None: # Append even if empty, to maintain consistency if needed later
            all_fwi_dfs.append(df_fwi_iter)
        


    print("Concatenating all iteration dataframes...")
    if all_fire_severity_dfs:
        final_fire_severity_df = pd.concat(all_fire_severity_dfs).drop_duplicates(subset=["Event_ID"])
    else:
        final_fire_severity_df = pd.DataFrame(columns=['Event_ID', 'fire_severity_mean', 'fire_severity_max'])

    if all_fwi_dfs:
        final_fwi_df = pd.concat(all_fwi_dfs)
        if 'time' in final_fwi_df.columns and not final_fwi_df.empty:
            final_fwi_df['time'] = pd.to_datetime(final_fwi_df['time'])
            final_fwi_df = final_fwi_df.drop_duplicates(subset=["Event_ID", "time"])
        elif 'Event_ID' in final_fwi_df.columns and not final_fwi_df.empty : # If no 'time' column but has Event_ID
             final_fwi_df = final_fwi_df.drop_duplicates(subset=["Event_ID"])
    else:
        final_fwi_df = pd.DataFrame(columns=['Event_ID', 'time', 'fwi_mean', 'fwi_max'])
        final_fwi_df['time'] = pd.to_datetime(final_fwi_df['time'])

    print("Merging FWI and Fire Severity data...")
    if not final_fwi_df.empty and not final_fire_severity_df.empty:
        fire_data_df = pd.merge(final_fwi_df, final_fire_severity_df, on="Event_ID", how="left")
    elif not final_fwi_df.empty:
        fire_data_df = final_fwi_df.copy()
        fire_data_df[['fire_severity_mean', 'fire_severity_max']] = pd.NA
    elif not final_fire_severity_df.empty:
        fire_data_df = final_fire_severity_df.copy()
        fire_data_df['time'] = pd.NaT
        fire_data_df[['fwi_mean', 'fwi_max']] = pd.NA
        fire_data_df['time'] = pd.to_datetime(fire_data_df['time'])
    else:
        fire_data_df = pd.DataFrame(columns=['Event_ID', 'time', 'fwi_mean', 'fwi_max', 'fire_severity_mean', 'fire_severity_max'])
        fire_data_df['time'] = pd.to_datetime(fire_data_df['time'])

    print("Merging with MTBS perimeter base data...")
    if 'time' in fire_data_df.columns:
        fire_data_df['time'] = pd.to_datetime(fire_data_df['time'])
    else: # Ensure time column exists for merge if fire_data_df was built without it
        fire_data_df['time'] = pd.NaT 
        fire_data_df['time'] = pd.to_datetime(fire_data_df['time'])
            
    mtbs_gdf_merged = pd.merge(
        mtbs_perimeters_gdf,
        fire_data_df,
        how='left',
        left_on=["Event_ID", "Ig_Date"],
        right_on=["Event_ID", "time"]
    )

    print("Loading and processing EIA service areas...")
    service_areas_raw = gpd.read_file(CONFIG["EIA_UTILITY_SERVICE_AREAS_PATH"])
    service_areas = service_areas_raw.to_crs(CONFIG["CRS"])

    print("Performing final selection, spatial join, and cleanup...")

    mtbs_gdf_final = mtbs_gdf_merged[["Event_ID", "Ig_Date", "BurnBndAc", "fire_severity_mean", "fire_severity_max", "fwi_mean", "fwi_max", "geometry"]]

    df_final = service_areas.sjoin(mtbs_gdf_final)
    df_final = df_final.drop(columns=["geometry", "index_right"])
    df_final = df_final.dropna(subset="fire_severity_mean")
    df_final = df_final.rename({"Event_ID": "Fire_ID"})
    df_final = df_final.sort_values(by=["ID", "Ig_Date"])

    print(f"Saving final data to {CONFIG['OUTPUT_CSV_PATH']}...")
    df_final.to_csv(CONFIG["OUTPUT_CSV_PATH"], index=False, compression="gzip")

    print("Script finished successfully.")
    print("Shutting down Dask client...")
    client.close()
    cluster.close()
    print("Dask client shut down.")

if __name__ == "__main__":
    main()