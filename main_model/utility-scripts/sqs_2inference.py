"""
sqs_2inference.py

Sending tile index to AWS SQS for a large scale inference
Tile inds files can be created from Geodex from AOI(s) geojson.

author: @DevelopmentSeed

usage:
python3 ultility-scripts/sqs_2inference.py --queue_url=https://sqs.us-east-1.amazonaws.com/xxx/MLInferenceUNICEFTestTileQueue \
                            --sqs_region=us-east-1 \
                            --tile_inds_txt=data/large_tile_inds.txt
"""

import os
from os import path as op
# import subprocess as subp

import boto3
from tqdm import tqdm

import argparse
import sys


def send_tiles(queue_url, sqs_region, tile_inds_txt):
    """
    Send tile inds to sqs for the inference
    -------
    Params
    queue_url: url from AWS sqs
    sqs_region: AWS sqs region
    tile_inds_txt: a single or multiple text files for tiles from geodex
    """
    message_body='{{"x":{x},"y":{y},"z":{z}}}'
    sqs = boto3.resource('sqs', region_name=sqs_region)
    queue = sqs.Queue(queue_url)
    with open(tile_inds_txt, 'r') as tile_file:
            tile_inds = [ind.strip() for ind in tile_file.readlines()]
    tile_inds = [ind.split(' ') for ind in tile_inds if len(ind)]
    print('Found {} tiles, pushing to SQS Queue: {}'.format(len(tile_inds),queue.url))
    for ti in tqdm(tile_inds):
        msg_body = message_body.format(x=ti[0], y=ti[1], z=ti[2])
        response = queue.send_message(MessageBody=msg_body,
                                      MessageAttributes={})
        if response.get('Failed'):
            print(response.get('Failed'))

def parse_arg(args):
    desc = "sqs_inference"
    dhf = argparse.RawTextHelpFormatter
    parse0 = argparse.ArgumentParser(description= desc, formatter_class=dhf)
    parse0.add_argument('--queue_url', help="url from AWS sqs")
    parse0.add_argument('--sqs_region', help='AWS sqs region')
    parse0.add_argument('--tile_inds_txt', help='a single or multiple text files for tiles from geodex')
    return vars(parse0.parse_args(args))

def main(queue_url, sqs_region, tile_inds_txt):
    send_tiles(queue_url, sqs_region, tile_inds_txt)

def cli():
    args = parse_arg(sys.argv[1:])
    main(**args)

if __name__ == "__main__":
    cli()
