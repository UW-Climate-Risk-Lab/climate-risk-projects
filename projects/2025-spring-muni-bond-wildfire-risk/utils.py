import pandas as pd
import xarray as xr
import requests # For downloading files
import os       # For path operations
import time     # For adding delays in retries
from typing import List, Optional

def _preprocess_add_time_from_filename(ds: xr.Dataset) -> xr.Dataset:
    """
    Preprocessing function for xarray.open_mfdataset.
    Extracts date from filename and assigns it as the time coordinate.
    Assumes filename format like: FWI.GPM.LATE.v5.Daily.Default.YYYYMMDD.nc
    """
    try:
        filename = os.path.basename(ds.encoding['source'])
        # Extract YYYYMMDD part. Example: FWI.GPM.LATE.v5.Daily.Default.20230315.nc
        date_str = filename.split('.')[-2] 
        
        if len(date_str) == 8 and date_str.isdigit():
            correct_time = pd.to_datetime(date_str, format='%Y%m%d')
            # Assign this timestamp to the 'time' coordinate.
            ds = ds.assign_coords(time=[correct_time])
        else:
            print(f"Warning: Could not parse date from filename: {filename}. Time coordinate will not be set for this file.")
    except Exception as e:
        filename_for_error = "unknown (ds.encoding not available)"
        if 'source' in ds.encoding:
            filename_for_error = ds.encoding['source']
        print(f"Warning: Error preprocessing file {filename_for_error} to add time coordinate: {e}")
    return ds

def download_and_read_fwi_data_for_dates(
    dates_series: pd.Series, 
    local_data_dir: str = "data/nccs_fwi_data",
    max_retries: int = 10,        # Maximum number of download retries
    retry_delay_seconds: int = 31 # Delay between retries
) -> Optional[xr.Dataset]:
    """
    Downloads NetCDF files from NASA NCCS portal to a local directory (if they 
    don't already exist), with retries for download failures. Then reads them 
    into a single xarray Dataset, adding a proper time coordinate during load.

    Args:
        dates_series (pd.Series): Pandas Series of datetime64 objects.
        local_data_dir (str): Local directory for NetCDF files.
        max_retries (int): Maximum number of times to retry a failed download.
        retry_delay_seconds (int): Seconds to wait between download retries.

    Returns:
        Optional[xr.Dataset]: Combined xarray Dataset or None on failure.
    """
    if not isinstance(dates_series, pd.Series):
        raise TypeError("Input 'dates_series' must be a pandas Series.")
    if dates_series.empty:
        print("Input dates_series is empty. Returning None.")
        return None

    try:
        os.makedirs(local_data_dir, exist_ok=True)
        print(f"Data will be handled in/downloaded to: {os.path.abspath(local_data_dir)}")
    except OSError as e:
        print(f"Error creating directory {local_data_dir}: {e}")
        return None

    base_url_template = "https://portal.nccs.nasa.gov/datashare/GlobalFWI/v2.0/fwiCalcs.GEOS-5/Default/GPM.LATE.v5/{year}/"
    filename_template = "FWI.GPM.LATE.v5.Daily.Default.{date_str}.nc"
    files_to_process_locally: List[str] = []

    for dt in dates_series:
        if not hasattr(dt, 'year') or not hasattr(dt, 'strftime'):
            try:
                dt = pd.to_datetime(dt)
            except ValueError as e:
                print(f"Warning: Could not convert '{dt}' to datetime: {e}. Skipping.")
                continue
        
        year = dt.year
        date_str = dt.strftime('%Y%m%d') 
        file_name = filename_template.format(date_str=date_str)
        local_target_path = os.path.join(local_data_dir, file_name)

        if os.path.exists(local_target_path):
            print(f"File '{file_name}' already exists locally. Using existing file.")
            files_to_process_locally.append(local_target_path)
            continue

        specific_base_url = base_url_template.format(year=year)
        full_url = specific_base_url + file_name
        
        # Download attempt loop with retries
        for attempt in range(max_retries):
            print(f"Attempting to download {full_url} (Attempt {attempt + 1}/{max_retries})...")
            try:
                response = requests.get(url=full_url, timeout=60) 
                response.raise_for_status() 
                
                with open(local_target_path, 'wb') as f:
                    f.write(response.content) 
                print(f"Successfully downloaded {file_name}")
                files_to_process_locally.append(local_target_path)
                del response
                break # Exit retry loop on success
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                print(f"Download error for {full_url} (Attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    print(f"Waiting {retry_delay_seconds} seconds before retrying...")
                    time.sleep(retry_delay_seconds)
                else:
                    print(f"Failed to download {full_url} after {max_retries} attempts.")
            except requests.exceptions.HTTPError as e: # Non-transient HTTP errors (e.g., 404 Not Found)
                print(f"HTTPError for {full_url}: {e}. Not retrying for this type of error.")
                break # Do not retry for HTTP errors like 404
            except requests.exceptions.RequestException as e: # Other general request errors
                print(f"General download error for {full_url}: {e}")
                # Decide if retry is appropriate for other RequestException types
                if attempt < max_retries - 1:
                    print(f"Waiting {retry_delay_seconds} seconds before retrying...")
                    time.sleep(retry_delay_seconds)
                else:
                    print(f"Failed to download {full_url} after {max_retries} attempts due to general error.")
                # break # Optionally break for all other request exceptions
            except IOError as e:
                print(f"IOError saving file {local_target_path}: {e}")
                break # Do not retry for IO errors

    if not files_to_process_locally:
        print("No files available for processing. Returning None.")
        return None

    print(f"\nAttempting to open {len(files_to_process_locally)} files with xarray, applying time preprocessing:")
    try:
        combined_ds = xr.open_mfdataset(
            files_to_process_locally, 
            combine='by_coords',
            preprocess=_preprocess_add_time_from_filename,
            engine='netcdf4' 
        )
        if 'time' in combined_ds.coords:
             combined_ds = combined_ds.sortby('time')
        return combined_ds
    except Exception as e:
        print(f"An error occurred while opening or combining local NetCDF files with xarray: {e}")
        return None

if __name__ == '__main__':
    data_directory = "my_fwi_data_downloads_v4" # New dir for testing retries
    max_download_retries = 3
    delay_between_retries = 10 # seconds

    # Example with a date that might cause issues or a mix of existing/new
    historical_dates_list = [
        '2014-05-14', # Should exist from previous run if using same parent dir structure
        '2014-05-13', # This one had a ConnectionResetError in your log
        '2014-08-14', # Successfully downloaded in your log
        '2014-08-13', # This one had a ConnectionResetError
        '2014-08-02', # This one had a ConnectionResetError
        '2014-07-30', # This one had a ConnectionResetError
        '2023-03-15'  # A different year, likely new
    ]
    my_dates_series = pd.Series(pd.to_datetime(historical_dates_list))

    print(f"--- Downloading and Reading FWI Data (with {max_download_retries} retries, {delay_between_retries}s delay) ---")
    print("Dates to process/fetch:")
    print(my_dates_series)
    print(f"Local directory for data: ./{data_directory}") 
    print("-" * 40)
    
    fwi_dataset = download_and_read_fwi_data_for_dates(
        my_dates_series, 
        local_data_dir=data_directory,
        max_retries=max_download_retries,
        retry_delay_seconds=delay_between_retries
    )

    if fwi_dataset:
        print("\n--- Successfully loaded dataset ---")
        print(fwi_dataset)
        if 'time' in fwi_dataset.coords:
            print("\nTime coordinate values:")
            print(fwi_dataset['time'].values)
            if pd.Series(fwi_dataset['time'].values).is_monotonic_increasing:
                print("Time coordinate is sorted.")
            else:
                print("Warning: Time coordinate is NOT sorted.")
        if 'FWI' in fwi_dataset.variables:
            print(f"\nFWI variable data type: {fwi_dataset['FWI'].dtype}")
        else:
            print("\n'FWI' variable not found. Available variables:", list(fwi_dataset.variables))
    else:
        print("\n--- Failed to load dataset or no data was successfully processed ---")

    print("\n--- Important Notes ---")
    print(f"Downloads attempted with up to {max_download_retries} retries and {delay_between_retries}s delay for connection issues.")
