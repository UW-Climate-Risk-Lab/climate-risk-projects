import os
import logging

import geopandas as gpd
import pandas as pd
import numpy as np
from shapely import Polygon
from shapely.geometry import box


MEDIAN_INCOME_DIRECTORY = "./data/zipcode_median_income"
ZIPCODE_SHAPE_FILE = "./data/zipcode_shape_file/tl_2020_us_zcta520.shp"
ELECTRIC_RETAIL_SERVICE_AREA_FILE = "./data/Electric_Retail_Service_Territories.geojson"
OUTPUT_FILE = "./data/utility_service_area_average_adjusted_gross_income.csv"

CRS = "EPSG:4269"
AREA_CRS ="EPSG:5070"
STATES = ["WA", "CA", "OR"]

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Roughly bounding box around California, Oregon, Washington.
# This is used to filter the zipcode geometries to only those within this region.
BBOX = gpd.GeoDataFrame(
    {'geometry': [box(-125.9, 31.9, -112.1, 50.1)]},
    crs='EPSG:4326'
).to_crs(CRS)


def load_median_income_dataframe() -> pd.DataFrame:
    """
    Reads and concatenates median income CSV files from the specified directory.

    Filters the data to include only the states of interest (WA, CA, OR) and adds a 'year' column
    based on the filename. Drops any rows with missing data.

    Returns:
        pd.DataFrame: A DataFrame containing the median income data with columns:
            - year (int): Year of the data.
            - zipcode (int): Zipcode number.
            - N1 (float): Number of tax returns.
            - A00100 (float): Total Adjusted Gross Income in thousands.
    """
    logging.info("Loading median income data...")
    data = []
    for file in os.listdir(MEDIAN_INCOME_DIRECTORY):
        if file.endswith(".csv"):
            df_temp = pd.read_csv(f"{MEDIAN_INCOME_DIRECTORY}/{file}")
            df_temp = df_temp.loc[df_temp["STATE"].isin(STATES)].copy()
            df_temp["year"] = int(f"20{file[0:2]}")
            data.append(df_temp[["year", "zipcode", "N1", "A00100"]].copy())
    df = pd.concat(data)
    df["zipcode"] = df["zipcode"].astype(int)
    df = df.dropna()
    logging.info("Loaded median income data.")
    return df


def load_zipcode_geodataframe() -> gpd.GeoDataFrame:
    """
    Loads and processes the geometries of each zipcode in the USA.

    Filters the geometries to only include those within the defined bounding box (BBOX)
    and converts the coordinate reference system to the specified CRS.

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame containing zipcode geometries with columns:
            - zipcode (int): Zipcode number.
            - geometry (Polygon): Geometry of the zipcode.
    """
    logging.info("Loading zipcode geodataframe...")
    columns_to_keep = ["ZCTA5CE20", "geometry"]

    gdf_zipcode_raw: gpd.GeoDataFrame = gpd.read_file(ZIPCODE_SHAPE_FILE)
    gdf_zipcode = gdf_zipcode_raw.to_crs(CRS)

    gdf_zipcode = gdf_zipcode.sjoin(BBOX)
    
    gdf_zipcode["ZCTA5CE20"] = gdf_zipcode["ZCTA5CE20"].astype(int)
    gdf_zipcode = gdf_zipcode.dropna()

    gdf_zipcode = (
        gdf_zipcode[columns_to_keep].copy().rename(columns={"ZCTA5CE20": "zipcode"})
    )

    logging.info("Loaded zipcode geodataframe.")
    return gdf_zipcode


def load_elec_service_area_geodataframe() -> gpd.GeoDataFrame:
    """
    Loads electric utility service area data from a GeoJSON file.

    Filters the data to include only the states of interest (WA, CA, OR) and renames columns
    for consistency. Converts the coordinate reference system to the specified CRS.

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame containing electric service area data with columns:
            - UTILITY_ID (str): Utility identifier.
            - NAME (str): Name of the utility.
            - geometry (Polygon): Geometry of the service area.
    """
    logging.info("Loading electric service area geodataframe...")
    columns = ["ID", "NAME", "geometry"]
    gdf_elec_service_areas_raw: gpd.GeoDataFrame = gpd.read_file(ELECTRIC_RETAIL_SERVICE_AREA_FILE)
    gdf_elec_service_areas = gdf_elec_service_areas_raw.loc[gdf_elec_service_areas_raw["STATE"].isin(STATES)].copy()
    gdf_elec_service_areas = gdf_elec_service_areas.to_crs(CRS)
    gdf_elec_service_areas = gdf_elec_service_areas[columns]
    gdf_elec_service_areas = gdf_elec_service_areas.rename(columns={"ID": "UTILITY_ID"})

    logging.info("Loaded electric service area geodataframe.")
    return gdf_elec_service_areas


def calc_zipcode_area_in_utility(
    gdf_zipcode: gpd.GeoDataFrame, gdf_elec_service_area: gpd.GeoDataFrame
) -> pd.DataFrame:
    """
    Calculates the percentage of each zipcode's area that overlaps with each electric utility service area.

    This is used to approximate the distribution of tax returns and adjusted gross income within utility service areas.

    Args:
        gdf_zipcode (gpd.GeoDataFrame): GeoDataFrame containing zipcode geometries.
        gdf_elec_service_area (gpd.GeoDataFrame): GeoDataFrame containing electric service area geometries.

    Returns:
        pd.DataFrame: A DataFrame with columns:
            - NAME (str): Name of the utility.
            - UTILITY_ID (str): Utility identifier.
            - zipcode (int): Zipcode number.
            - percent_of_zipcode_in_service_area (float): Percentage of the zipcode's area within the service area.
    """
    logging.info("Calculating zipcode area in utility service areas...")
    gdf_zipcode_polygons= gdf_zipcode.explode()
    gdf_zipcode_polygons["zipcode_area"] = gdf_zipcode_polygons["geometry"].to_crs(AREA_CRS).geometry.area
    gdf_elec_utility_polygons = gdf_elec_service_area.explode()

    gdf_intersection = gpd.overlay(
        gdf_zipcode_polygons, gdf_elec_utility_polygons, how="intersection", keep_geom_type=True
    )
    gdf_intersection = gdf_intersection.loc[gdf_intersection['zipcode_area'] > 1e-9].copy()

    gdf_intersection['intersection_area'] = gdf_intersection["geometry"].to_crs(AREA_CRS).geometry.area
    gdf_intersection['percent_of_zipcode_in_service_area'] = gdf_intersection['intersection_area'] / gdf_intersection['zipcode_area']
    gdf_intersection['percent_of_zipcode_in_service_area'] = gdf_intersection['percent_of_zipcode_in_service_area'].clip(0, 1)
    gdf_intersection = gdf_intersection.loc[gdf_intersection["percent_of_zipcode_in_service_area"] > 0].copy()

    df_intersection = pd.DataFrame(gdf_intersection[["NAME", "UTILITY_ID", "zipcode", "percent_of_zipcode_in_service_area"]])

    logging.info("Calculated zipcode area in utility service areas.")
    return df_intersection


def main():
    """
    Main function to execute the data processing workflow.

    Loads zipcode and electric service area data, calculates the area overlap,
    and merges this with median income data to compute a weighted average adjusted gross income
    for each utility area. The results are saved to a CSV file.
    """
    logging.info("Starting main process...")
    gdf_zipcode = load_zipcode_geodataframe()
    gdf_elec_service_area = load_elec_service_area_geodataframe()
    df_service_area = calc_zipcode_area_in_utility(gdf_zipcode, gdf_elec_service_area)
    df_median_income = load_median_income_dataframe()

    df = df_median_income.merge(df_service_area, how="left", on="zipcode")
    
    # Here, we do the final calculation, which is a weighted average adjusted gross income for each utility area
    # We do this by summing the total adjusted gross income dollars (AGI) and dividing by the total number of returns
    # We approximate the total number of returns and AGI in a given utility area by multiplying through the percent
    # of the zipcode that overlaps that utility service area with the total AGI and total number of returns for that given zipcode.
    # This helps gives a better APPROXIMATION for a given electric utilty's actual AGI.

    df["approximate_utility_tax_returns"] = df["percent_of_zipcode_in_service_area"] * df["N1"] 
    df["approximate_utility_total_agi"] = df["A00100"] * 1000 * df["percent_of_zipcode_in_service_area"] # Units: dollars
    s = df.groupby(["NAME", "year"]).apply(
        lambda x: sum(x["approximate_utility_total_agi"]) / sum(x["approximate_utility_tax_returns"]), include_groups=False)
    
    s.name = "average_adjusted_gross_income"
    df_final = pd.DataFrame(s)

    logging.info("Saving final dataframe to CSV...")
    df_final.to_csv(OUTPUT_FILE)
    logging.info("Process completed and data saved.")


if __name__ == "__main__":
    main()
