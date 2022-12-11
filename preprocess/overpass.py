from tqdm import tqdm
import requests

# Set the base URL for the Overpass API
overpass_url = "http://overpass-api.de/api/interpreter"

# Set the query string to retrieve all drivable roads in the given region
# Replace "BBOX" with the bounding box coordinates of the region
query_string = """
[out:json];
area[name="-76.14703928120434,40.613771673487776,-74.38973296433687,39.531704791809375"];
(way["highway"]["access"!~"private|no"]["motor_vehicle"!~"no"](area);
way["motorcar"]["access"!~"private|no"](area);
way["motor_vehicle"]["access"!~"private|no"](area);
way["vehicle"]["access"!~"private|no"](area);
);
out body;
>;
out skel qt;
"""

# Send the HTTP request to the Overpass API
response = requests.get(overpass_url, params={'data': query_string})

# Parse the JSON response
data = response.json()

# Print the number of roads in the response
print("Found {} roads".format(len(data['elements'])))

# Print the names and coordinates of each road
for element in tqdm(data['elements']):
    if 'name' in element['tags']:
        print("{}: {}".format(element['tags']['name'], element['geometry']))
    else:
        print("Unnamed road: {}".format(element['geometry']))