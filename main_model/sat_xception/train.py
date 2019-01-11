import os
from os import path as op
from functools import partial
import numpy as np
from sat_xception.utils import (print_start_details, print_end_details)
from datetime import datetime as dt
from sat_xception.train_config import (tboard_dir, ckpt_dir, data_dir,
                    model_params as MP, train_params as TP, data_flow as DF)
from hyperopt import fmin, Trials, STATUS_OK, tpe
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.xception import Xception, preprocess_input as xcept_preproc
from sat_xception.xception import xcept_net
from sat_xception.mobilenetv2 import mobilenet_v2
from hyperopt import hp
import pickle

def get_params(MP, TP):
    """Return hyperopt parameters"""
    return dict(
        optimizer=hp.choice('optimizer', MP['optimizer']),
        lr_phase1=hp.uniform('lr_phase1', MP['lr_phase1'][0], MP['lr_phase1'][1]),
        lr_phase2=hp.uniform('lr_phase2', MP['lr_phase2'][0], MP['lr_phase2'][1]),
        weight_init=hp.choice('weight_init', MP['weight_init']),
        freeze_cutoff=hp.choice('freeze_cutoff', MP['freeze_cutoff']),
        dropout_rate=hp.choice('dropout_rate', MP['dropout_rate']),
        dense_size=hp.choice('dense_size', MP['dense_size']),
        dense_activation=hp.choice('dense_activation', MP['dense_activation']),
        n_epo_phase1=hp.quniform('n_epo_phase1', TP['n_epo_phase1'][0], TP['n_epo_phase1'][1], 1),
        #n_epo_phase2=hp.quniform('n_epo_phase2', TP['n_epo_phase2'][0], TP['n_epo_phase2'][1], 1),
        n_epo_phase2=TP['n_epo_phase2'],
        n_classes=MP['n_classes'],
        output_activation=hp.choice('output_activation', MP['output_activation']),
        loss=hp.choice('loss', MP['loss']),
        steps_per_train_epo=TP['steps_per_train_epo'],
        steps_per_test_epo=TP['steps_per_test_epo'],
        max_queue_size=TP['max_queue_size'],
        workers=TP['workers'],
        use_multiprocessing=TP['use_multiprocessing'],
        class_weight=TP['class_weight'])

def train(train_dir, validation_dir, model_id):
    start_time = dt.now()
    print_start_details(start_time)



    ###################################
    # Calculate number of train/test images
    ###################################
    total_test_images = 0
    # Print out how many images are available for train/test
    for fold in [train_dir, validation_dir]:
        for sub_fold in ['not_school', 'school']:
            temp_img_dir = op.join(data_dir, fold, sub_fold)
            n_fnames = len([fname for fname in os.listdir(temp_img_dir)
                            if op.splitext(fname)[-1] in ['.png', '.jpg']])
            print('For {}ing, found {} {} images'.format(fold, n_fnames, sub_fold))

            if fold == validation_dir:
                total_test_images += n_fnames
    if TP['steps_per_test_epo'] is None:
        TP['steps_per_test_epo'] = np.ceil(total_test_images / DF['flow_from_dir']['batch_size']) + 1

    ###################################
    # Set up generators
    ###################################
    # train_gen = ImageDataGenerator(preprocessing_function=xcept_preproc,
    #                                **DF['image_data_generator'])
    # test_gen = ImageDataGenerator(preprocessing_function=xcept_preproc)
    if model_id=="xception":
        model = xcept_net
    else:
        model = mobilenet_v2

    trials = Trials()
    algo = partial(tpe.suggest, n_startup_jobs=5)
    argmin = fmin(model, space=get_params(MP, TP), algo=algo,
                  max_evals=100, trials=trials)

    end_time = dt.now()
    print_end_details(start_time, end_time)
    print("Evalutation of best performing model:")
    print(trials.best_trial['result']['loss'])

    with open(op.join(ckpt_dir, 'trials_{}.pkl'.format(start_time)), "wb") as pkl_file:
        pickle.dump(trials, pkl_file)
