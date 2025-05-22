import xarray as xr
import os
import numpy as np
import rioxarray
from rasterio.enums import Resampling
import statsmodels.api as sm
from pathlib import Path
import logging
import zarr
import fsspec
import s3fs

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def load_datasets(s3_bucket):
    """Load all required datasets"""
    logger.info("Loading datasets...")
    
    # Load burn probability and flame length exceedance datasets
    ds_burn_probability = xr.open_dataset("data/BP_WA.tif", engine="rasterio")
    
    # Load historical FWI dataset
    ds_fwi_historical = xr.open_dataset(
        f"s3://{s3_bucket}/climate-risk-map/backend/climate/scenariomip/NEX-GDDP-CMIP6/DECADE_MONTH_ENSEMBLE/historical/fwi_decade_month_historical.zarr", 
        engine="zarr"
    )
    
    # Load future FWI dataset (2030)
    ds_fwi_2030 = xr.open_dataset(
        f"s3://{s3_bucket}/climate-risk-map/backend/climate/scenariomip/NEX-GDDP-CMIP6/DECADE_MONTH_ENSEMBLE/ssp370/fwi_decade_month_ssp370.zarr", 
        engine="zarr"
    )
    
    return ds_burn_probability, ds_fwi_historical, ds_fwi_2030

def filter_fire_season_months(ds, months=['05', '06', '07', '08', '09', '10']):
    """Filter dataset to include only fire season months (May-October)"""
    month_part = np.array([month_str.split('-')[1] for month_str in ds.decade_month.values])
    month_mask = np.isin(month_part, months)
    return ds.isel(decade_month=month_mask)

def calculate_mean_fwi_by_month(ds):
    """Calculate mean FWI across years for each month"""
    logger.info("Calculating mean FWI by month...")
    
    # Extract month from decade_month
    ds = ds.assign_coords(
        month=("decade_month", [dm.split('-')[1] for dm in ds.decade_month.values])
    )
    
    # Calculate mean across years for each month
    monthly_mean_fwi = ds.groupby("month").mean()
    
    return monthly_mean_fwi

def calculate_future_probability(p_now, fwi_now, fwi_future):
    """Calculate future probability using relative change method"""
        
    # Calculate relative change (1.0 means no change)
    relative_change = 1.0 + (fwi_future - fwi_now) / fwi_future
    p_future = p_now * relative_change
    
    # Ensure probability is between 0 and 1
    p_future = np.clip(p_future, 0, 1)
    
    return p_future

def main():
    """Main execution function"""
    s3_bucket = os.environ.get("S3_BUCKET")
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)
    
    # Load datasets
    ds_burn_probability, ds_fwi_historical, ds_fwi_2030 = load_datasets(s3_bucket)
    
    # Filter to fire season months (May-October)
    ds_fwi_historical = filter_fire_season_months(ds_fwi_historical)
    ds_fwi_2030 = filter_fire_season_months(ds_fwi_2030)
    
    # Calculate historical mean FWI by month
    ds_fwi_historical = calculate_mean_fwi_by_month(ds_fwi_historical)
    ds_fwi_2030 = calculate_mean_fwi_by_month(ds_fwi_2030)

    # Reproject FWI data to match burn probability spatial resolution and extent
    logger.info("Reprojecting FWI data to match burn probability data...")
    da_burn_probability = ds_burn_probability.sel(band=1)["band_data"]
    da_burn_probability = da_burn_probability.rio.reproject("EPSG:4326")
    da_burn_probability = da_burn_probability.rio.reproject("EPSG:4326")
    
    fwi_historical_reproj = ds_fwi_historical['value_q3'].rio.write_crs("EPSG:4326")
    fwi_historical_reproj.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
    fwi_historical_reproj = fwi_historical_reproj.rio.reproject_match(da_burn_probability, resampling=Resampling.bilinear)
    
    fwi_2030_reproj = ds_fwi_2030['value_q3'].rio.write_crs("EPSG:4326")
    fwi_2030_reproj.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
    fwi_2030_reproj = fwi_2030_reproj.rio.reproject_match(da_burn_probability, resampling=Resampling.bilinear)
    
    
    # Calculate future burn probabilities for each month
    logger.info("Calculating future burn probabilities...")
    future_burn_probability = []
    
    for month in ds_fwi_2030.month.values:

        # Calculate future probabilities
        _da_future_burn_probability = calculate_future_probability(
            da_burn_probability, 
            fwi_historical_reproj.sel(month=month), 
            fwi_2030_reproj.sel(month=month),
        )
        
        _da_future_burn_probability = _da_future_burn_probability.assign_coords(month=month)
        
        future_burn_probability.append(_da_future_burn_probability)
    
    da_burn_probability_future = (xr.concat(future_burn_probability, dim='month'))

    fwi_historical_reproj = fwi_historical_reproj
    fwi_2030_reproj = fwi_2030_reproj

    # Combine monthly future probabilities
    ds_burn_prob = xr.Dataset({
        'burn_probability_future_2030': da_burn_probability_future,
        'burn_probability_current': da_burn_probability,
        'fwi_current': fwi_historical_reproj,
        'fwi_future_2030': fwi_2030_reproj
    })
    
    # Save results to zarr files
    logger.info("Saving results to zarr files...")
    s3_output_uri = f"s3://{s3_bucket}/student-projects/amazon-wildfire-risk-spring2025/data/cmip6_adjusted_burn_probability.zarr"
    # Let to_zarr() handle the computation
    fs = s3fs.S3FileSystem(
                anon=False,
                )
    # Let to_zarr() handle the computation
    ds_burn_prob.to_zarr(
        store=s3fs.S3Map(root=s3_output_uri, s3=fs),
        mode="w",
        consolidated=True,
    )
    
    logger.info("Processing complete!")

if __name__ == "__main__":
    main()