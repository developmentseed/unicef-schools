"""
pred_2geojson.py

Script to covert ML predicted schools result in CSV into geojosn tile polygon or points.


author: @DevelopmentSeed

usage:
python3 pred_2geojson.py --csv_file=results_unicef_4.csv \
                         --keyword=polygon \
                         --threshold=0.92


"""

import sys, os
import csv
import json

from os import path as op
from mercantile import ul, feature, Tile
from geojson import Feature, FeatureCollection

from shapely.geometry import shape
from tqdm import tqdm
import argparse
import sys

def pred_2geojson(csv_file, keyword, threshold):
    features =  list()
    with open(csv_file, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in tqdm(reader):
            pred = json.loads(row[1])
            pred_red = list(map(lambda x: round(x, 2), pred))
            pred_obj = dict(zip(map(lambda x: 'p%s' % x, range(len(pred_red))), pred_red))
            if keyword =="point":
                pred_j = ul(*[int(t) for t in row[0].split('-')])
                feature_collection =Feature(geometry=dict(type='Point', coordinates=[pred_j.lng, pred_j.lat]),
                                    properties=pred_obj)
            else:
                pred_j = feature(Tile(*[int(t) for t in row[0].split('-')]))
                feature_ =Feature(geometry=pred_j['geometry'],
                                            properties=pred_obj)
            if pred_obj['p1'] >=float(threshold):
                features.append(feature_)
    feature_collection = FeatureCollection(features)
    with open('results_{}_{}.geojson'.format(keyword, threshold), 'w') as results:
        json.dump(feature_collection, results)

def parse_arg(args):
    desc = "pred_2geojson"
    dhf = argparse.RawTextHelpFormatter
    parse0 = argparse.ArgumentParser(description= desc, formatter_class=dhf)
    parse0.add_argument('--csv_file', help="path to csv that saves final predictions")
    parse0.add_argument('--keyword', help='use point or polygon')
    parse0.add_argument('--threshold', help='set the thresold e.g 0.85')
    return vars(parse0.parse_args(args))

def main(csv_file, keyword, threshold):
    pred_2geojson(csv_file, keyword, threshold)

def cli():
    args = parse_arg(sys.argv[1:])
    main(**args)

if __name__ == "__main__":
    cli()
