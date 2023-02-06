import geopandas as gpd
import pandas as pd
import numpy as np
import networkx as nx
import osmnx as ox
import shapely as shp
import json
from pathlib import Path
from tqdm import tqdm
import time
import fire

OG_CSV_PATH = Path("/deep/u/ayushsn/satellite-pixel-synthesis-pytorch/preprocess/all_negatives_combined.csv")
DVRPC_BOUNDARY_PATH = Path("/deep/u/ayushsn/aicc-spr22-transportation/preprocess/util_scripts/dvrpc_boundary.shp")
ALL_DRIVE = (
	f'["highway"]["area"!~"yes"]'
	f'["highway"!~"abandoned|bridleway|construction|corridor|cycleway|elevator|planned|escalator|'
	f'footway|path|pedestrian|platform|proposed|raceway|steps|track"]'
    f'["motor_vehicle"!~"no"]["motorcar"!~"no"]'
    f'["service"!~"parking|private"]'
)

# drive+service: allow ways tagged 'service' but filter out certain types
drive_service = (
    f'["highway"]["area"!~"yes"]["access"!~"private"]'
    f'["highway"!~"abandoned|bridleway|bus_guideway|construction|corridor|cycleway|elevator|'
    f'escalator|footway|path|pedestrian|planned|platform|proposed|raceway|steps|track"]'
    f'["motor_vehicle"!~"no"]["motorcar"!~"no"]'
    f'["service"!~"emergency_access|parking|parking_aisle|private"]'
)

def filter(mode="all_drive"):
    print(f"OSMNX is filtering using mode {mode}.")
    print(f"Loading paths from {OG_CSV_PATH.name}...")
    train = pd.read_csv(OG_CSV_PATH, header=0)
    image_paths = np.asarray(train.iloc[:, 1])
    print("Loaded Image Paths.")

    print("Loading OSMNX stuff...")
    start_time = time.time()
    dvrpc_boundary_gpd = gpd.read_file(DVRPC_BOUNDARY_PATH)
    dvrpc_boundary = dvrpc_boundary_gpd.geometry[0]
    if mode == "all_drive":
        G = ox.graph_from_polygon(dvrpc_boundary, custom_filter = ALL_DRIVE)
    else:
        G = ox.graph_from_polygon(dvrpc_boundary, network_type = mode)
    end_time = time.time()
    print(f"Time OSMNX stuff took: {(end_time - start_time) / 60} minutes.")

    has_roads = []
    to_label = {True: "yes", False: "no"}
    for idx, image_path in enumerate(tqdm(image_paths)):
        with open(Path(image_path).with_suffix('.json')) as json_file:
            data = json.load(json_file)
        polygon = shp.wkt.loads(str(data['extent']))

        try:
            truncatedG = ox.truncate.truncate_graph_polygon(G, polygon, truncate_by_edge=True)
            has_road = not nx.is_empty(truncatedG)
        except ValueError:
            has_road = False

        has_roads.append(to_label[has_road])

    df = pd.DataFrame({"image": image_paths, "has_road": has_roads})
    df.to_csv(OG_CSV_PATH.with_stem(f"{mode}_no_private_diff"))

if __name__ == "__main__":
    fire.Fire()
