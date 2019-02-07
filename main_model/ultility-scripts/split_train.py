"""
split_train.py

cript to split training dataset to train and test in 70:20:10

author: @DevelopmentSeed

usage:

python3 split_train.py \
       --train_dir=school_1m_zoom18 \
       --keyword=school \
       --img_format='.jpg'

The above python script will split train, validation and test for school tiles
You can split the train and test in the same way e.g for hospital or other catagories

"""


import os
from os import makedirs, path as op
import glob
import random
from random import shuffle
import shutil
import argparse
import sys

def split_train(train_dir, keyword, img_format = ".jpg"):

    """
    script to split train and test dataset for image classification by 80:20

    :param train_dir: path to rgb training dataset
    :param keyword: catagory name as the keyword
    :param img_format: the image format that produced from tile serve

    :return None: split train and validation in seperate directories with the keywords as the sub-directories.
    """

    train_path = op.join(op.abspath(op.join(train_dir, os.pardir)), "train")
    val_path = op.join(op.abspath(op.join(train_dir, os.pardir)), "validation")
    test_path = op.join(op.abspath(op.join(train_dir, os.pardir)), "test")
    if not op.isdir(train_path):
        makedirs(train_path)
    if not op.isdir(val_path):
        makedirs(val_path)
    if not op.isdir(test_path):
        makedirs(test_path)
    img_train_path = op.join(train_path, keyword)
    if not op.isdir(img_train_path):
        makedirs(img_train_path)
    img_val_path = op.join(val_path, keyword)
    if not op.isdir(img_val_path):
        makedirs(img_val_path)
    img_test_path = op.join(test_path, keyword)
    if not op.isdir(img_test_path):
        makedirs(img_test_path)
    images = sorted(glob.glob(train_dir + "/*" + img_format))
    random.seed(230)
    shuffle(images)

    split_train = int(0.7 * len(images))
    split_valid = int(0.9 * len(images))
    # split_test = int(len(images)-split_valid)
    print("total training, validationg and test dataset are")
    train_files = images[:split_train]
    val_files = images[split_train:split_valid]
    test_files = images[split_valid:]
    print("{}, {} and {}".format(len(train_files),len(val_files) ,len(test_files)))
    for img in train_files:
        shutil.copy(img, img_train_path)
    for img_v in val_files:
        shutil.copy(img_v, img_val_path)
    for img_t in test_files:
        shutil.copy(img_t, img_test_path)

def parse_arg(args):
    desc = "train_data_split"
    dhf = argparse.RawTextHelpFormatter
    parse0 = argparse.ArgumentParser(description= desc, formatter_class=dhf)
    parse0.add_argument('--train_dir', help="path to all the dataset need to seperate into train and test")
    parse0.add_argument('--keyword', help='catagory name as the keywork to save images')
    parse0.add_argument('--img_format', help='image format among tif, jpg, or png')
    return vars(parse0.parse_args(args))

def main(train_dir,keyword, img_format):
    split_train(train_dir, keyword, img_format = ".jpg")

def cli():
    args = parse_arg(sys.argv[1:])
    main(**args)




if __name__ == "__main__":
    cli()
