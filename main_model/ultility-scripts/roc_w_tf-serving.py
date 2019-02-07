"""
roc_w_tf-servimg.py

Sending RGB image tiles to a running tf-serving POST for prediction and plotting ROC curve

author: @DevelopmentSeed

usage:
python3 roc_w_tf-servimg.py --test_path=test \
        --keyword1=not_school \
        --keyword2=school \
        --server_endpoint='http://localhost:8501/v1/models/2nd-iter_more-schools_tf-serving:predict' \
        --plot_dir=plot_dir \
        --model_time="0122_071829"


"""

# needed package for serving the tf-serving image
import os
import json
import pprint
import time
import base64
import requests
import glob
import itertools

# needed package for plotting ROC curve

from os import makedirs, path as op
import numpy as np
from sklearn.metrics import roc_curve, auc
from scipy.spatial.distance import euclidean
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
from keras import backend as K
from keras.applications.xception import preprocess_input as xcept_preproc
import yaml
from tqdm import tqdm

import argparse
import sys

def grouper(n, iterable, fillvalue=None):
    "grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return itertools.zip_longest(*args, fillvalue=fillvalue)

def prediction(test_path, keyword1, keyword2, server_endpoint):
    """
    function to compare y_test and y_predction in order to plot the ROC curve

    ----
    Params:
    test_path: file path for the test dataset
    keyword1: the catagory name 1 under the test path
    keyword2: the catagory name 2 under the test path
    server_endpoint: the url of tf-serving endpoint when you run your tf-serving docker image

    ----
    return: y_test and y_predict as numpy array
    """
    not_school_images_paths  = sorted(glob.glob(op.join(test_path, keyword1) + "/*.jpg"))
    school_images_paths = sorted(glob.glob(op.join(test_path, keyword2) + "/*.jpg"))

    y_test = list()
    for img in school_images_paths:
        y_test.append([img, 1, 0])

    for img in not_school_images_paths:
        y_test.append([img, 0, 1])

    imgs_lst = [item[0] for item in y_test]
    y_pred = []
    for group in grouper(50, imgs_lst):
        instances = []
        for img_fpath in group:
            print(img_fpath)
            try:
                with open(img_fpath, 'rb') as imageFile:
                    b64_image = base64.b64encode(imageFile.read())
                    instances.append({'image_bytes': {'b64': b64_image.decode('utf-8')}})
            except:
                pass

        payload = json.dumps({"instances": instances})

        start = time.time()
        r = requests.post(server_endpoint, data=payload)
        elapsed = time.time() - start

        #########################
        # Print results
        #########################
        pp = pprint.PrettyPrinter()
        print('\nPredictions from local images:')
        pp.pprint(json.loads(r.content)['predictions'])
        y_pred.append(json.loads(r.content)['predictions'])
    #     y_pred = [group] + y_pred

        print('Elapsed time: {} sec'.format(elapsed))
    y_pred_flatten = [item for sublist in y_pred for item in sublist]
    print("the total images go to plot ROC curve is:")
    len_2plot = len(y_pred_flatten)
    print("*" * 40)
    print(len_2plot)
    # get the first column of the y_test with the first 3000
    y_test_2plot = [item[2] for item in y_test[:len_2plot]]
    # get the first prediction column  of y_pred for school
    y_pred_2plot = [item[0] for item in y_pred_flatten]
    y_test_arr = np.array(y_test_2plot)
    y_pred_arr = np.array(y_pred_2plot)

    return y_test_arr, y_pred_arr

def plot_roc(y_test_arr, y_pred_arr, plot_dir, model_time):
    """
    Plot ROC curve
    """
    if not op.isdir(plot_dir):
        makedirs(plot_dir)
    plot_dir = plot_dir
    model_time = model_time

    y_neg_pred = y_pred_arr[y_test_arr == 0]  # Predictions for negative examples
    y_pos_pred = y_pred_arr[y_test_arr == 1]  # Predictions for positive examples

    # Accuracy (should match tensorboard)
    correct = np.sum(y_test_arr == np.round(y_pred_arr))
    total = y_test_arr.shape[0]
    acc = float(correct)/ float(total)
    print('Accuracy: {:0.5f}'.format(acc))
    # Compute FPR, TPR for '1' label (i.e., positive examples)
    fpr, tpr, thresh = roc_curve(y_test_arr, y_pred_arr)
    roc_auc = auc(fpr, tpr)

    # Min corner dist (*one* optimal value for threshold derived from ROC curve)
    corner_dists = np.empty((fpr.shape[0]))
    for di, (x_val, y_val) in enumerate(zip(fpr, tpr)):
        corner_dists[di] = euclidean([0., 1.], [x_val, y_val])
    opt_cutoff_ind = np.argmin(corner_dists)
    min_corner_x = fpr[opt_cutoff_ind]
    min_corner_y = tpr[opt_cutoff_ind]

    ####################
    # Plot
    ####################
    print('Plotting.')
    plt.close('all')
    sns.set()
    sns.set_style('darkgrid', {"axes.facecolor": ".9"})
    sns.set_context('talk', font_scale=1.1)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.plot(fpr, tpr, lw=2, label='ROC curve (area={:0.2f})'.format(roc_auc))
    ax.plot([min_corner_x, min_corner_x], [0, min_corner_y],
            color='r', lw=1, label='Min-corner distance\n(FPR={:0.2f}, thresh={:0.2f})'.format(min_corner_x, thresh[opt_cutoff_ind]))
    plt.plot([0, 1], [0, 1], color='black', lw=0.75, linestyle='--')
    ax.set_xlim([-0.03, 1.0])
    ax.set_ylim([0.0, 1.03])
    ax.set_xlabel('False Positive Rate\n(1 - Specificity)')
    ax.set_ylabel('True Positive Rate\n(Sensitivity)')
    ax.set_aspect('equal')
    ax.set_title('ROC curve for schools detection')
    plt.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(op.join(plot_dir, 'roc_{}.png'.format(model_time)),
                dpi=150)

    # Plot a kernel density estimate and rug plot
    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 6))
    kde_kws = dict(shade=True, clip=[0., 1.], alpha=0.3)
    rug_kws = dict(alpha=0.2)
    sns.distplot(y_neg_pred, hist=False, kde=True, rug=True, norm_hist=True, color="b",
                 kde_kws=kde_kws, rug_kws=rug_kws, label='True negatives', ax=ax2)
    sns.distplot(y_pos_pred, hist=False, kde=True, rug=True, norm_hist=True, color="r",
                 kde_kws=kde_kws, rug_kws=rug_kws, label='True positives', ax=ax2)
    ax2.set_title('Predicted scores for true positives and true negatives')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_xlabel("Model's predicted score")
    ax2.set_ylabel('Probability density')
    plt.legend(loc="best")
    fig2.savefig(op.join(plot_dir, 'dist_fpr_tpr_{}.png'.format(model_time)),
                 dpi=150)

def parse_arg(args):
    desc = "plot_ROC_tf-serving"
    dhf = argparse.RawTextHelpFormatter
    parse0 = argparse.ArgumentParser(description= desc, formatter_class=dhf)
    parse0.add_argument('--test_path', help="file path for the test dataset")
    parse0.add_argument('--keyword1', help='the catagory name 1 under the test path')
    parse0.add_argument('--keyword2', help='the catagory name 2 under the test path')
    parse0.add_argument('--server_endpoint', help='the url of tf-serving endpoint when you run your tf-serving docker image')
    parse0.add_argument('--plot_dir', help='the directory to save plotted ROC curve')
    parse0.add_argument('--model_time', help='The model time from a trained model')
    return vars(parse0.parse_args(args))

def main(test_path, keyword1, keyword2, server_endpoint, plot_dir, model_time):
    y_test_arr, y_pred_arr = prediction(test_path, keyword1, keyword2, server_endpoint)
    plot_roc(y_test_arr, y_pred_arr, plot_dir, model_time)

def cli():
    args = parse_arg(sys.argv[1:])
    main(**args)

if __name__ == "__main__":
    cli()
