"""
download_tiles.py

You need to parse a tile-grid geojson file here
And the tile-grid (from points to tiles) can be done using Development Seed GeoKit tool.

author: @DevelopmentSeed

usage:

python3 download_tiles.py --tile_geojson=2nd_yes_cleaned_school_tiles.geojson \
                          --tile_url='https://a.tiles.mapbox.com/v4/digitalglobe.2lnpeioh/{z}/{x}/{y}.jpg?access_token=TOKEN'


The above python script will split train, validation and test for school tiles
You can split the train and test in the same way e.g for hospital or other catagories


"""

import json
try:
    from urllib.parse import urlparse, parse_qs
except ImportError:
     from urlparse import urlparse, parse_qs
import requests
import os
from os import makedirs, path as op
import argparse
import sys

def get_tile(geojson, base_url):
    """
    Function to download tiles for school and not-school.
    The tile index was created using DevSeed Geokit with 1m buffer to the geolocation points for school and not-school classes;
    :param geojson: geojson for tile and tile index from geokit (poin2tile);
    :param base_url: url to access DG vivid and given the token to download the tiles.

    :return tiles: a list of tiles
    """
    # open geojson and get tile index
    with open(geojson, 'r') as data:
        tile_geojson = json.load(data)
    features = tile_geojson["features"]
    # get the tile index as x, y, z formats.
    xyz = [features[i]['properties']['tiles'] for i in range(len(features))]

    # create tile folder
    tiles_folder = op.splitext(geojson)[0]
    if not op.isdir(tiles_folder):
        makedirs(tiles_folder)

    # download and get the list of tiles
    tiles = list()
    for i in range(len(xyz)):
        x=str(xyz[i][0])
        y=str(xyz[i][1])
        z=str(xyz[i][2])
        url = base_url.replace('{x}', x).replace('{y}', y).replace('{z}', z)
        o = urlparse(url)
        _, image_format = op.splitext(o.path)
        tile_bn ="{}-{}-{}{}".format(z, x, y,image_format)
        r = requests.get(url)
        tile= op.join(tiles_folder, tile_bn)
        tiles.append(tile)
        with open(tile, 'wb')as w:
            w.write(r.content)
    return tiles


def parse_arg(args):
    desc = "download_tiles"
    dhf = argparse.RawTextHelpFormatter
    parse0 = argparse.ArgumentParser(description= desc, formatter_class=dhf)
    parse0.add_argument('--tile_geojson', help="a geojson tile-grid for the desired tile zoom level")
    parse0.add_argument('--tile_url', help='web map tile service url')
    return vars(parse0.parse_args(args))

def main(tile_url, tile_geojson):
    get_tile(tile_geojson, tile_url)

def cli():
    args = parse_arg(sys.argv[1:])
    main(**args)

if __name__ == "__main__":
    cli()
