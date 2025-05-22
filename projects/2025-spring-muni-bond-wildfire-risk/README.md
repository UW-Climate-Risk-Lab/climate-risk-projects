# Muni Bond × Wildfire Risk Analysis

This repository provides tools to calculate fire severity and Fire Weather Index (FWI) metrics for municipal utility service areas.  
It combines MTBS fire severity rasters, MTBS perimeter shapefiles, EIA utility service area boundaries, and NASA NCCS FWI data.

## Contents
- `main.py` — orchestrates data loading, processing, zonal statistics, and final merge.
- `utils.py` — helper functions for downloading FWI data and adding proper time coordinates.
- `pyproject.toml` — project metadata and dependencies.
- `data/` — expected location for input raster, perimeter, and service‐area datasets.
- `iteration_cache/` — intermediate CSV caches for speed on re‐runs.
- `utility_service_area_fire_metrics.csv.gz` — final output.

## Data Sources

1. **MTBS Raster Data**  
   - Located in `./data/mtbs_rasters/`  
   - Annual burn severity rasters (GeoTIFF) from the Monitoring Trends in Burn Severity (MTBS) program.

2. **MTBS Perimeter Shapefiles**  
   - Located in `./data/mtbs_perimeters/mtbs_perims_DD.shp`  
   - Provides fire perimeters with ignition dates (`Ig_Date`) and `Event_ID`.

3. **EIA Utility Service Areas**  
   - Located in `./data/eia_service_areas/Electric_Retail_Service_Territories.geojson`  
   - Boundaries for electric service territories from the U.S. Energy Information Administration.

4. **FWI (Fire Weather Index)**  
   - Downloaded on‐the‐fly from NASA NCCS:  
     `https://portal.nccs.nasa.gov/datashare/GlobalFWI/v2.0/fwiCalcs.GEOS-5/Default/GPM.LATE.v5/{year}/FWI.GPM.LATE.v5.Daily.Default.{YYYYMMDD}.nc`  
   - Stored locally in `data/nccs_fwi_data/` or custom directory via config.

> **Note:** The above datasets are hosted by the Climate Risk Lab and are available upon request.

## Installation

1. Clone the repo  
   ```bash
   git clone https://github.com/UW-Climate-Risk-Lab/climate-risk-map.git
   cd muni_bond_spring_2025
   ```

2. Assuming UV is installed, you can run the script using the following 
   ```bash
   uv sync
   uv run main.py
   ```

> **Note:** Other dependencies include GDAL and GEOS libraries on your machine.

## Configuration

Edit the top of `main.py` (the `CONFIG` dict) to adjust:
- Paths to input data (`MTBS_RASTER_DATA_DIR`, `MTBS_PERIMETER_DATA_PATH`, `EIA_UTILITY_SERVICE_AREAS_PATH`)
- Dask settings (`N_WORKERS`, `MEMORY_PER_WORKER`, etc.)
- FWI variable name and spatial slices
- Output file path

## Running the Script

The script will:
1. Launch a local Dask cluster.
2. Load and reproject MTBS perimeters and rasters.
3. Download FWI files for each ignition date (with retry logic).
4. Compute zonal mean & max for fire severity and FWI.
5. Merge results with service areas and save to:

```
utility_service_area_fire_metrics.csv.gz
```

## Output

- **`utility_service_area_fire_metrics.csv.gz`**  
  Column summary:
  - `Fire_ID` (Event_ID)
  - `Ig_Date` (ignition date)
  - `BurnBndAc` (burned acreage)
  - `fire_severity_mean`, `fire_severity_max`
  - `fwi_mean`, `fwi_max`
  - `ID` (service area identifier)

## Requesting Data

All raw datasets (MTBS rasters, perimeters, service areas) are maintained by the Climate Risk Lab.  
Please contact the lab administrators to obtain download links or direct access.

