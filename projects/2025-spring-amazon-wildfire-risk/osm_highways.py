import overpy
import json
import time
import sys
from datetime import datetime

def process_north_spokane_corridor(api):
    """
    Special handling for North Spokane Corridor which might use different tags.
    """
    queries = [
        # Search by name
        """
        area["name"="Washington"]["admin_level"="4"]->.searchArea;
        (
          way["highway"]["name"~"North Spokane Corridor|NSC|Spokane North Corridor"](area.searchArea);
        );
        (._;>;);
        out body;
        """,
        
        # Search by description or alt_name
        """
        area["name"="Washington"]["admin_level"="4"]->.searchArea;
        (
          way["highway"]["description"~"North Spokane Corridor|NSC"](area.searchArea);
          way["highway"]["alt_name"~"North Spokane Corridor|NSC"](area.searchArea);
        );
        (._;>;);
        out body;
        """,
        
        # Search for US-395 sections in Spokane
        """
        area["name"="Spokane"]["admin_level"~"8|6"]->.spokane;
        (
          way["highway"]["ref"~"US 395|US-395|395"](area.spokane);
          way["highway"]["ref:us"="395"](area.spokane);
        );
        (._;>;);
        out body;
        """
    ]
    
    all_ways = []
    for i, query in enumerate(queries):
        try:
            print(f"Trying NSC query approach {i+1}...")
            result = api.query(query)
            print(f"Found {len(result.ways)} ways with approach {i+1}")
            all_ways.extend(result.ways)
            time.sleep(2)  # Respect rate limits
        except Exception as e:
            print(f"Error with NSC query approach {i+1}: {e}")
            time.sleep(5)
    
    # Remove duplicates by ID
    unique_ways = {way.id: way for way in all_ways}
    print(f"Found {len(unique_ways)} unique ways for North Spokane Corridor")
    return list(unique_ways.values())

def save_to_geojson(ways, filename):
    """
    Save highway data to GeoJSON with complete LineString geometry
    """
    if not ways:
        print("No data to save.")
        return
        
    features = []
    for way in ways:
        # Skip ways with fewer than 2 nodes (can't form a proper LineString)
        if len(way.nodes) < 2:
            continue
            
        # Create the complete coordinate array for the LineString
        # Each coordinate is [longitude, latitude] as per GeoJSON spec
        coordinates = [[float(node.lon), float(node.lat)] for node in way.nodes]
        
        # Prepare properties object with all tags and metadata
        properties = {
            'id': way.id,
            'osm_type': 'way',
            'node_count': len(way.nodes),
            'node_ids': [node.id for node in way.nodes]
        }
        
        # Add all OSM tags to properties
        properties.update(way.tags)
        
        # Create the feature with full LineString geometry
        feature = {
            'type': 'Feature',
            'geometry': {
                'type': 'LineString',
                'coordinates': coordinates
            },
            'properties': properties
        }
        
        features.append(feature)
    
    # Create the final GeoJSON object
    geojson = {
        'type': 'FeatureCollection',
        'features': features,
        'metadata': {
            'generated': datetime.now().isoformat(),
            'count': len(features),
            'description': 'Washington State highway segments extracted from OpenStreetMap'
        }
    }
    
    # Write to file with pretty formatting
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(geojson, f, indent=2)
        
    print(f"GeoJSON data saved to {filename}")
    print(f"Successfully exported {len(features)} highway segments with complete geometry")

def main():
    print("OpenStreetMap Washington Highway Extractor")
    print("=========================================")
    
    # Create timestamp for file naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        api = overpy.Overpass()
    except NameError:
        print("Error: overpy module not found. Please install it with 'pip install overpy'.")
        sys.exit(1)
        
    all_ways = []
    
    # Handle North Spokane Corridor separately
    print("\nFetching data for North Spokane Corridor...")
    nsc_ways = process_north_spokane_corridor(api)
    all_ways.extend(nsc_ways)
    
    # Remove duplicates
    unique_ways = {way.id: way for way in all_ways}
    unique_ways_list = list(unique_ways.values())
    
    if not unique_ways_list:
        print("\nNo data found. Please check your internet connection and try again.")
        sys.exit(1)
    
    # Create filename
    filename = f"data/highways/north_spokane_corridor.geojson"
    
    # Save data in GeoJSON format with complete LineString geometries
    save_to_geojson(unique_ways_list, filename)
    
    print(f"\nComplete highway data has been saved to {filename}")
    print(f"The GeoJSON file contains {len(unique_ways_list)} highway segments")
    print("Each segment includes full LineString geometry and all OSM tags")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        sys.exit(1)