import geopandas as gpd
import pandas as pd
import numpy as np
import networkx as nx
import osmnx as ox
import shapely as shp

dvrpc_boundary_gpd = gpd.read_file("/deep/u/ayushsn/aicc-spr22-transportation/preprocess/util_scripts/dvrpc_boundary.shp")
dvrpc_boundary = dvrpc_boundary_gpd.geometry[0]
G = ox.graph_from_polygon(dvrpc_boundary, network_type = 'all_private')
#G_projected = ox.project_graph(G)
#G_consl = ox.consolidate_intersections(G_projected, rebuild_graph = True, tolerance = 15, dead_ends = False)
ox.save_graph_geopackage(G, filepath='/deep/u/ayushsn/satellite-pixel-synthesis-pytorch/preprocess/dvrpc_osm_G.gpkg')
ox.save_graph_shapefile(G, filepath='/deep/u/ayushsn/satellite-pixel-synthesis-pytorch/preprocess/graph_shapefiles_G')
