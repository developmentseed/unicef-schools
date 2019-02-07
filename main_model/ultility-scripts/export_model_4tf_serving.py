"""
export_model_4tf_serving.py

Script to export trained model to create tensorflow serving docker image.

author: @developmentseed

usage:

python3 export_model_4tf_serving.py \
        --model_weight z18-xception_models/0122_071829_L0.28_E10_weights.h5 \
        --model_arc z18-xception_models/0122_071829_arch.yaml \
        --model_params models/0122_071829_params.yaml \
        --export_path unicef_school_tf_serving \
        --model_dir z18-xception_models
"""

import os
from os import makedirs, path as op

import tensorflow as tf
import keras
from keras.models import model_from_yaml

from keras.layers.core import K
import yaml
import sys
import argparse



def load_model(model_fpath, weights_fpath):
    """Load a model from yaml architecture and h5 weights."""
    assert model_fpath[-5:] == '.yaml'
    assert weights_fpath[-3:] == '.h5'

    with open(model_fpath, "r") as yaml_file:
        yaml_architecture = yaml_file.read()

    model = model_from_yaml(yaml_architecture)
    model.load_weights(weights_fpath)

    return model

def serving_input_receiver_fn():
    """Convert b64 string  encoded images into tensor"""
    HEIGHT, WIDTH, CHANNELS = 256, 256, 3

    def decode_and_resize(image_str_tensor):
        """Decodes jpeg string, resizes it and returns a uint8 tensor."""
        image = tf.image.decode_image(image_str_tensor, channels=CHANNELS,
                                    dtype=tf.uint8)
        image = tf.reshape(image, [HEIGHT, WIDTH, CHANNELS])

        return image

    # Run processing for batch prediction.
    input_ph = tf.placeholder(tf.string, shape=[None], name='image_binary')
    images_tensor = tf.map_fn(
        decode_and_resize, input_ph, back_prop=False, dtype=tf.uint8)

    # Cast to float and run xception preprocessing on images (to scale [0, 255] to [-1, 1])
    images_tensor = tf.cast(images_tensor, dtype=tf.float32)
    images_tensor = tf.subtract(tf.divide(images_tensor, 127.5), 1)

    return tf.estimator.export.ServingInputReceiver(
        {'input_1': images_tensor},
        {'image_bytes': input_ph})


def parse_args(args):
    desc = "tf-serving_image"
    dhf = argparse.ArgumentDefaultsHelpFormatter
    parse0 = argparse.ArgumentParser(description= desc, formatter_class= dhf)
    parse0.add_argument('--model_weight', help='path to saved h5 model weight')
    parse0.add_argument('--model_arc', help='path to saved yaml file of model architecture')
    parse0.add_argument('--model_params', help='path to save yaml file of the model parameters')
    parse0.add_argument('--export_path', help='path for exporting model to')
    parse0.add_argument('--model_dir', help='path to saved model weigh, architecture and parameters')
    return vars(parse0.parse_args(args))

def main(model_weight, model_arc, model_params, export_path, model_dir):

    if not op.isdir(export_path):
        os.mkdir(export_path)

    model = load_model(model_arc,model_weight)
    # Compile model (necessary for creating an estimator). However, no training will be done here
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'])
    model.save(model_weight)

    tf_model = tf.keras.models.load_model(model_weight)

    estimator = tf.keras.estimator.model_to_estimator(keras_model=tf_model,
                                                      model_dir=model_dir)

    estimator.export_savedmodel(
        op.join(export_path),
        serving_input_receiver_fn=serving_input_receiver_fn)


def cli():
    args = parse_args(sys.argv[1:])
    main(**args)

if __name__ == "__main__":
    cli()
