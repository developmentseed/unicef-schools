
"""
train_config.py

List some configuration parameters for training model
"""

import os
from os import path as op


# Set directories for saving model weights and tensorboard information
data_dir = os.getcwd()

#     cloud_comp = False

ckpt_dir = op.join(os.getcwd(), "models")
tboard_dir = op.join(os.getcwd(), "tensorboard")
preds_dir = op.join(os.getcwd(), "preds")
plot_dir = op.join(os.getcwd(), "plots")
cloud_comp = False

if not op.isdir(ckpt_dir):
    os.mkdir(ckpt_dir)
if not op.isdir(tboard_dir):
    os.mkdir(tboard_dir)

model_params = dict(loss=['binary_crossentropy'], #use binary crossentropy
                    optimizer=[dict(opt_func='adam'),
                               dict(opt_func='rmsprop')],
                               # SGD as below performed notably poorer in 1st big hyperopt run
                               #dict(opt_func='sgd', momentum=hp.uniform('momentum', 0.5, 0.9))],
                    lr_phase1=[1e-4, 1e-3],  # learning rate for phase 1 (output layer only)
                    lr_phase2=[1e-5, 1e-4],  # learning rate for phase 2 (all layers beyond freeze_cutoff)
                    weight_init=['glorot_uniform'],
                    metrics=['accuracy'], # change the evaluation metrics from val_categorical_accuracy to just accuracy
                    # Blocks organized in 10s, 66, 76, 86, etc.
                    freeze_cutoff=[0],  # Layer below which no training/updating occurs on weights
                    dense_size=[128, 256, 512],  # Number of nodes in 2nd to final layer
                    n_classes=2,  # Number of class choices in final layer
                    output_activation=['softmax'], #using sigmoid seems lower the accuracy
                    dense_activation=['relu', 'elu'],
                    dropout_rate=[0, 0.1, 0.25, 0.5])  # Dropout in final layer

train_params = dict(n_rand_hp_iters=5,
                    n_total_hp_iters=100,  # Total number of HyperParam experiments to run
                    n_epo_phase1=[1, 2],  # Number of epochs training only top layer
                    n_epo_phase2=15,  # Number of epochs fine tuning whole model
                    max_queue_size=128,
                    workers=8,
                    use_multiprocessing=False,
                    #prop_total_img_set=0.5,  # Proportion of total images per train epoch
                    img_size=(256, 256, 3),
                    early_stopping_patience=5,  # Number of iters w/out val_acc increase
                    early_stopping_min_delta=0.01,
                    reduce_lr_patience=3,  # Number of iters w/out val_acc increase
                    reduce_lr_min_delta=0.1,
                    class_weight={0: 2, 1: 1},  # Based on pakistan_redux image counts
                    steps_per_train_epo=256,
                    steps_per_test_epo=None)

# Define params for ImageDataGenerator and ImageDataGenerator.flow_from_directory
data_flow = dict(image_data_generator=dict(horizontal_flip=True,
                                           vertical_flip=True,
                                           rotation_range=180,
                                           zoom_range=(1, 1.2),
                                           brightness_range=(0.8, 1.2),
                                           channel_shift_range=10),
                 flow_from_dir=dict(target_size=train_params['img_size'][:2],  # Only want width/height here
                                    color_mode='rgb',
                                    classes=['not_school', 'school'],  # Keep this ordering, it should match class_weights
                                    batch_size=32,  # Want as large as GPU can handle, using batch-norm layers
                                    seed=42,  # Seed for random number generator
                                    save_to_dir=None))  # Set to visualize augmentations


# data flow for mobilenetv2, the image size has to be (224, 224, 3) instead of default (256, 256, 3)
data_flow_mn = dict(image_data_generator=dict(horizontal_flip=True,
                                           vertical_flip=True,
                                           rotation_range=180,
                                           zoom_range=(1, 1.2),
                                           brightness_range=(0.8, 1.2),
                                           channel_shift_range=10),
                 flow_from_dir=dict(target_size=(224, 224),  # Only want width/height here
                                    color_mode='rgb',
                                    classes=['not_school', 'school'],  # Keep this ordering, it should match class_weights
                                    batch_size=32,  # Want as large as GPU can handle, using batch-norm layers
                                    seed=42,  # Seed for random number generator
                                    save_to_dir=None))
