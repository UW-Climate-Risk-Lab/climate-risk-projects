"""
This script uses the overpass API to query locations that
likely belong to Amazon's supply chain. The output is a CSV 
file parsed for relevant facilities for the analysis.

"""


import overpy
import json
from datetime import datetime
import pandas as pd


def find_amazon_facilities_in_washington():
    """
    Find all Amazon supply chain, distribution, fulfillment, and logistics facilities
    in Washington state using the Overpass API.
    """
    # Initialize the Overpass API
    api = overpy.Overpass()

    # Define the EASTERN Washington state bounding box
    # Format: south latitude, west longitude, north latitude, east longitude
    wa_bbox = "(45.6,-121.5,49.0,-116.94)"

    # Define Amazon-specific facility naming patterns
    amazon_name_pattern = (
        "Amazon.*[Ff]ulfillment|Amazon.*[Dd]elivery|Amazon.*[Dd]istribution|"
        "Amazon.*[Ss]ort|Amazon.*[Ww]arehouse|Amazon.*[Ll]ogistics|Amazon.*[Aa]ir.*[Hh]ub|"
        "Amazon.*HUB|Amazon.*[Hh]ub"
    )

    # Build the query for Overpass API - more focused on confirmed Amazon facilities
    query = f"""
    [out:json][timeout:90];
    (
      // Search by operator=Amazon tag (most reliable indicator)
      node["operator"="Amazon"]{wa_bbox};
      way["operator"="Amazon"]{wa_bbox};
      relation["operator"="Amazon"]{wa_bbox};
      
      // Search by Amazon-specific name patterns
      node["name"~"{amazon_name_pattern}"]{wa_bbox};
      way["name"~"{amazon_name_pattern}"]{wa_bbox};
      relation["name"~"{amazon_name_pattern}"]{wa_bbox};
      
      // Search by brand=Amazon
      node["brand"="Amazon"]{wa_bbox};
      way["brand"="Amazon"]{wa_bbox};
      relation["brand"="Amazon"]{wa_bbox};
      
      // Search by Amazon's correct Wikidata ID
      node["brand:wikidata"="Q3884"]{wa_bbox};
      way["brand:wikidata"="Q3884"]{wa_bbox};
      relation["brand:wikidata"="Q3884"]{wa_bbox};
      
      // Search for warehouses with specific Amazon facility codes (BFI, DSE, etc.)
      node["ref"~"^(BFI|DSE|DWA|DSW|PDX|GEG|SEA|PAE|UWA|DPD|HBF|HWA|DWX|HGE|DSK|KGE|RNT|OLM|PSC)[0-9]+"]{wa_bbox};
      way["ref"~"^(BFI|DSE|DWA|DSW|PDX|GEG|SEA|PAE|UWA|DPD|HBF|HWA|DWX|HGE|DSK|KGE|RNT|OLM|PSC)[0-9]+"]{wa_bbox};
      relation["ref"~"^(BFI|DSE|DWA|DSW|PDX|GEG|SEA|PAE|UWA|DPD|HBF|HWA|DWX|HGE|DSK|KGE|RNT|OLM|PSC)[0-9]+"]{wa_bbox};
    );
    // Output the basic elements
    out body;
    // Include all nodes that make up ways and relations
    >;
    // Output remaining elements
    out skel qt;
    """

    # Execute query
    print("Executing Overpass query for Amazon facilities in Washington...")
    result = api.query(query)

    # Process the results
    facilities = {"nodes": [], "ways": [], "relations": []}

    # Process nodes
    print(f"Found {len(result.nodes)} nodes")
    for node in result.nodes:
        facilities["nodes"].append(
            {
                "id": node.id,
                "lat": float(node.lat),
                "lon": float(node.lon),
                "tags": node.tags,
            }
        )

    # Process ways
    print(f"Found {len(result.ways)} ways")
    for way in result.ways:
        facilities["ways"].append(
            {
                "id": way.id,
                "nodes": [node.id for node in way.nodes],
                "tags": way.tags,
                "center": (
                    {
                        "lat": sum(float(n.lat) for n in way.nodes) / len(way.nodes),
                        "lon": sum(float(n.lon) for n in way.nodes) / len(way.nodes),
                    }
                    if way.nodes
                    else None
                ),
            }
        )

    # Process relations
    print(f"Found {len(result.relations)} relations")
    for relation in result.relations:
        facilities["relations"].append(
            {
                "id": relation.id,
                "members": [
                    {"type": member.role, "ref": member.ref}
                    for member in relation.members
                ],
                "tags": relation.tags,
            }
        )

    # Categorize facilities - only Amazon facilities
    categorized = categorize_facilities(facilities)

    return categorized


def categorize_facilities(facilities):
    """
    Categorize facilities based on their tags and names,
    ensuring they are actually Amazon facilities.
    """
    categories = {
        "fulfillment_centers": [],
        "delivery_stations": [],
        "sort_centers": [],
        "distribution_centers": [],
        "warehouses": [],
        "logistics_centers": [],
        "air_hubs": [],
        "other": [],
    }

    # Helper function to determine if a facility is Amazon-related
    def is_amazon_facility(tags):
        # Check for definitive Amazon identifiers
        if (
            tags.get("operator") == "Amazon"
            or tags.get("brand") == "Amazon"
            or tags.get("brand:wikidata") == "Q3884"
        ):
            return True

        # Check if name specifically mentions Amazon
        name = tags.get("name", "").lower()
        if name and "amazon" in name:
            return True

        # Check for Amazon facility codes in ref tag
        ref = tags.get("ref", "").upper()
        if ref and any(
            ref.startswith(prefix)
            for prefix in [
                "BFI",
                "DSE",
                "DWA",
                "DSW",
                "PDX",
                "GEG",
                "PAE",
                "UWA",
                "HBF",
                "HWA",
                "DWX",
                "HGE",
                "DSK",
                "KGE",
                "RNT",
                "OLM",
                "PSC",
                "SEA",
            ]
        ):
            return True

        return False

    # Helper function to categorize a facility
    def categorize(facility, facility_type):
        tags = facility.get("tags", {})

        # Skip non-Amazon facilities
        if not is_amazon_facility(tags):
            return

        name = tags.get("name", "").lower() if tags else ""
        ref = tags.get("ref", "").upper() if tags else ""

        facility_info = {"type": facility_type, "id": facility["id"], "tags": tags}

        # Add coordinates if available
        if facility_type == "node":
            facility_info["coordinates"] = {
                "lat": facility["lat"],
                "lon": facility["lon"],
            }
        elif facility_type == "way" and "center" in facility:
            facility_info["center"] = facility["center"]

        # Categorize based on facility tags and codes
        if (
            "fulfillment" in name
            or (ref and ref.startswith("BFI"))
            or (ref and ref.startswith("PDX"))
            or (ref and ref.startswith("GEG"))
        ):
            categories["fulfillment_centers"].append(facility_info)
        elif (
            "delivery station" in name
            or "delivery center" in name
            or (ref and (ref.startswith("DSE") or ref.startswith("DSW")))
        ):
            categories["delivery_stations"].append(facility_info)
        elif "sort center" in name or (ref and ref.startswith("SC")):
            categories["sort_centers"].append(facility_info)
        elif "distribution" in name and "amazon" in name:
            categories["distribution_centers"].append(facility_info)
        elif "warehouse" in tags.get("building", "") or "warehouse" in name:
            categories["warehouses"].append(facility_info)
        elif "logistics" in name:
            categories["logistics_centers"].append(facility_info)
        elif (
            "air" in name and ("hub" in name or "freight" in name)
        ) or "air hub" in name:
            categories["air_hubs"].append(facility_info)
        else:
            # Only include in "other" if it's confirmed to be Amazon
            categories["other"].append(facility_info)

    # Categorize all facilities
    for node in facilities["nodes"]:
        categorize(node, "node")

    for way in facilities["ways"]:
        categorize(way, "way")

    for relation in facilities["relations"]:
        categorize(relation, "relation")

    # Add summary statistics
    categories["summary"] = {
        "total": sum(
            len(items)
            for category, items in categories.items()
            if category != "summary"
        ),
        "by_category": {
            category: len(items)
            for category, items in categories.items()
            if category != "summary"
        },
    }

    return categories


def convert_to_dataframe(categorized_data):
    """
    Convert the categorized data to a pandas DataFrame for CSV export.
    """
    rows = []

    # Process each category
    for category, facilities in categorized_data.items():
        if category == "summary":
            continue

        for facility in facilities:
            # Extract common data
            facility_type = facility.get("type")
            facility_id = facility.get("id")
            tags = facility.get("tags", {})

            # Extract coordinates
            lat = lon = None
            if "coordinates" in facility:
                lat = facility["coordinates"].get("lat")
                lon = facility["coordinates"].get("lon")
            elif "center" in facility:
                lat = facility["center"].get("lat")
                lon = facility["center"].get("lon")

            # Extract common tags
            name = tags.get("name", "Unnamed")
            ref = tags.get("ref", "")
            operator = tags.get("operator", "")
            building_type = tags.get("building", "")
            addr_city = tags.get("addr:city", "")
            addr_state = tags.get("addr:state", "")
            addr_street = tags.get("addr:street", "")
            addr_housenumber = tags.get("addr:housenumber", "")
            addr_postcode = tags.get("addr:postcode", "")

            # Create a row
            row = {
                "category": category,
                "osm_type": facility_type,
                "osm_id": facility_id,
                "name": name,
                "ref": ref,
                "operator": operator,
                "building_type": building_type,
                "latitude": lat,
                "longitude": lon,
                "full_address": f"{addr_housenumber} {addr_street}, {addr_city}, {addr_state} {addr_postcode}".strip(
                    ", "
                ),
                "metadata": json.dumps(
                    tags
                ),  # Add all tags as a JSON string in a metadata column
            }

            rows.append(row)

    # Create DataFrame
    df = pd.DataFrame(rows)

    # Sort by category and name
    df = df.sort_values(by=["category", "name"])

    df = df_cleaning_steps(df=df)

    return df

def df_cleaning_steps(df: pd.DataFrame):
    """Manual cleaning steps to remove unwanted
    facilities and parse attributes

    Args:
        df (pd.DataFrame): Dataframe
    """

    df_clean = df[(~df["latitude"].isnull())].copy()
    df_clean = df_clean[(~df_clean["longitude"].isnull())]
    df_clean = df_clean[~(df_clean["name"].str.contains("Road"))]

    return df_clean



def save_results_to_csv(results):
    """Save results to a CSV file with a timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"amazon_facilities_eastern_washington.csv"

    # Convert to DataFrame
    df = convert_to_dataframe(results)

    # Save to CSV
    df.to_csv(f"data/{filename}", index=False)

    print(f"Results saved to {filename}")
    return filename

def print_summary(results):
    """Print a summary of the results"""
    summary = results["summary"]

    print("\n===== AMAZON FACILITIES IN WASHINGTON STATE =====")
    print(f"Total facilities found: {summary['total']}")
    print("\nBreakdown by category:")
    for category, count in summary["by_category"].items():
        print(f"  {category.replace('_', ' ').title()}: {count}")


# Main execution
if __name__ == "__main__":
    try:
        results = find_amazon_facilities_in_washington()
        print_summary(results)

        # Save results to CSV
        csv_filename = save_results_to_csv(results)
        print(f"\nTabular results saved to {csv_filename}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
