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

def cleanup(suffix=3000):
    print(f"Loading paths from train_road_{suffix}.csv")
    train = pd.read_csv(f'/deep/u/ayushsn/satellite-pixel-synthesis-pytorch/preprocess/train_road_{suffix}.csv', header=0)
    image_paths = np.asarray(train.iloc[:, 1])
    print("Loaded Image Paths...")

    print("Loading OSMNX stuff...")
    start_time = time.time()
    dvrpc_boundary_gpd = gpd.read_file("/deep/u/ayushsn/aicc-spr22-transportation/preprocess/util_scripts/dvrpc_boundary.shp")
    dvrpc_boundary = dvrpc_boundary_gpd.geometry[0]
    G = ox.graph_from_polygon(dvrpc_boundary, network_type = 'all_private')
    end_time = time.time()
    print(f"Time OSMNX stuff took: {(end_time - start_time) / 60} minutes")

    has_roads = []
    for idx, image_path in enumerate(tqdm(image_paths)):
        with open(Path(image_path).with_suffix('.json')) as json_file:
            data = json.load(json_file)
        polygon = shp.wkt.loads(str(data['extent']))

        try:
            truncatedG = ox.truncate.truncate_graph_polygon(G, polygon, truncate_by_edge=True)
            has_road = not nx.is_empty(truncatedG)
        except ValueError:
            has_road = False

        has_roads.append(has_road)

    df = pd.DataFrame({"image": image_paths, "has_road": has_roads})
    df.to_csv(f"/deep/u/ayushsn/satellite-pixel-synthesis-pytorch/preprocess/train_road_labeled_{suffix}.csv")

if __name__ == "__main__":
    fire.Fire()