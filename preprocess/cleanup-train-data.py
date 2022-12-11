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

start_time = time.time()
dvrpc_boundary_gpd = gpd.read_file("/deep/u/ayushsn/aicc-spr22-transportation/preprocess/util_scripts/dvrpc_boundary.shp")
dvrpc_boundary = dvrpc_boundary_gpd.geometry[0]
G = ox.graph_from_polygon(dvrpc_boundary, network_type = 'all_private')
##G_projected = ox.project_graph(G)
##G_consl = ox.consolidate_intersections(G_projected, rebuild_graph = True, tolerance = 15, dead_ends = False)
end_time = time.time()
print(f"Time OSMNX stuff took: {(end_time - start_time) / 60} minutes")

#gdf = gpd.read_file("/deep/u/ayushsn/satellite-pixel-synthesis-pytorch/preprocess/graph_shapefiles_G/nodes.shp")
#gdf = gpd.read_file("/deep/u/ayushsn/satellite-pixel-synthesis-pytorch/preprocess/dvrpc_osm_G.gpkg", layer='edges')
#unioned = gdf.unary_union

# train = pd.read_csv("/deep/group/aicc-bootcamp/transportation/data/fusion/train.csv", header=0)
# image_paths = np.asarray(train.iloc[:, 1])
image_paths = [
    "/deep/group/aicc-bootcamp/transportation/data/dvrpc/aerial/good2010/imagery/500:0:0.1:18:-303:89579.png",  # no sidewalks
    "/deep/group/aicc-bootcamp/transportation/data/dvrpc/aerial/good2015/imagery/500:0:0.1:18:-244:88534.png",  # w/ sidewalks at an intersection 
    "/deep/group/aicc-bootcamp/transportation/data/dvrpc/aerial/good2015/imagery/500:0:0.1:18:-655:88899.png",  # only sidewalks, no intersection
]
# more positive examples here - /deep/group/aicc-bootcamp/transportation/data/dvrpc/labels/examples

rows_to_drop = []
for idx, image_path in enumerate(tqdm(image_paths)):
    with open(Path(image_path).with_suffix('.json')) as json_file:
        data = json.load(json_file)
    polygon = shp.wkt.loads(str(data['extent']))
    
    try:
        truncatedG = ox.truncate.truncate_graph_polygon(G, polygon, truncate_by_edge=True)
        to_keep = not nx.is_empty(truncatedG)
    except ValueError:
        to_keep = False
    
    #to_keep = gdf.intersects(polygon).any()
    #to_keep = unioned.intersects(polygon)
    
    print(to_keep)  # expected: False, True, True
    
    # if not to_keep:
    #     rows_to_drop.append(idx)

# train = train.drop(rows_to_drop)
# train.save_csv("/deep/u/ayushsn/satellite-pixel-synthesis-pytorch/preprocess/train_osmnx.csv")