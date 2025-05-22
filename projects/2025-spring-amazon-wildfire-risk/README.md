# MBA Amazon Wildfire Risk Analysis (Spring 2025)

This project aims to assess wildfire risks in Eastern Washington, focusing on Amazon facilities and their exposure to fire hazards. The analysis integrates geospatial data, climate projections, and infrastructure information to provide actionable insights.

## Methodology

### 1. **Amazon Facility Identification**
   - **Script**: `osm_amazon_locations.py`
   - **Objective**: Identify Amazon facilities in Eastern Washington using the Overpass API.
   - **Steps**:
     1. Query OpenStreetMap data for nodes, ways, and relations matching Amazon-specific tags (e.g., `operator=Amazon`, `brand=Amazon`).
     2. Filter results to include only confirmed Amazon facilities based on naming patterns, tags, and facility codes.
     3. Categorize facilities into types such as fulfillment centers, delivery stations, warehouses, etc.
     4. Save the results as a CSV file for further analysis.

### 2. **Wildfire Risk Modeling**
   - **Script**: `fire_probability.py`
   - **Objective**: Calculate current and future wildfire probabilities using climate data.
   - **Steps**:
     1. Load burn probability data (USDA) and Fire Weather Index (FWI) datasets (derived from NASA NEX CMIP6) for historical and future scenarios.
     2. Filter FWI data to focus on fire season months (May–October).
     3. Calculate mean FWI by month for both historical and future datasets.
     4. Reproject FWI data to match the spatial resolution of burn probability data.
     5. Compute future burn probabilities by adjusting current probabilities based on relative changes in FWI.
     6. Save the results as a Zarr dataset for efficient storage and access.

### 3. **Exposure Analysis**
   - **Script**: `exposure_calc.py`
   - **Objective**: Assess the exposure of Amazon facilities to wildfire risks and nearby infrastructure.
   - **Steps**:
     1. **Point-Based Analysis**:
        - Extract wildfire risk metrics (e.g., burn probability, FWI) at facility locations.
     2. **Buffer-Based Analysis**:
        - Create buffers (5, 10, 25 miles) around facilities.
        - Calculate zonal statistics (e.g., max burn probability) within each buffer.
     3. **Infrastructure Metrics**:
        - Calculate distances to the nearest fire stations and electrical substations.
        - Count the number of substations within each buffer radius.
     4. Combine all results into a comprehensive dataset and save as an Excel file.

### 4. **Data Integration and Outputs**
   - The scripts integrate geospatial data from OpenStreetMap, climate projections from CMIP6, and infrastructure data (fire stations, substations).
   - Outputs include:
     - CSV files with categorized Amazon facilities.
     - Zarr datasets with adjusted burn probabilities.
     - Excel files summarizing wildfire exposure and infrastructure metrics.

## Data Sources
- **OpenStreetMap**: Facility locations and attributes.
- **CMIP6 Climate Projections**: Historical and future Fire Weather Index data.
- **Local Infrastructure Data**: Fire stations and electrical substations.

## Usage
1. Run `osm_amazon_locations.py` to generate the facility dataset.
2. Use `fire_probability.py` to calculate wildfire probabilities.
3. Execute `exposure_calc.py` to analyze exposure and generate the final report.

## Dependencies
- Python libraries: `overpy`, `pandas`, `xarray`, `rioxarray`, `geopandas`, `shapely`, `s3fs`, `statsmodels`, `openpyxl`, `zarr`
- External tools: Overpass API, S3-compatible storage for climate data.

## Results
The analysis provides:
- A detailed breakdown of wildfire risks for Amazon facilities.
- Insights into the proximity of critical infrastructure.
- Data-driven recommendations for mitigating wildfire risks.

