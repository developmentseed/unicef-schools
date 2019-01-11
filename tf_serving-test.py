
import os
from os import makedirs, path as op

import tensorflow as tf
import keras
from keras.models import model_from_yaml

from tensorflow.python.saved_model import (signature_constants, tag_constants,
                                           builder)
from tensorflow.python.saved_model.signature_def_utils import predict_signature_def
from keras.layers.core import K


def load_keras_model(arch_fpath, weights_fpath, n_gpus=0):
    """Load a Keras architecture and weights from osm_task_metrics
    Parameters
    ----------
    arch_fpath: str
        Architecture saved as YAML file
    weights_fpath: str
        Weights saved as h5 file
    n_gpus: int
        Number of gpus available to run prediction. Default 0.
    Returns
    -------
    model: keras.models.Model
    """

    if not op.splitext(arch_fpath)[-1] == '.yaml':
        raise ValueError('Model filepath must have `.yaml` extension.')
    if not op.splitext(weights_fpath)[-1] == '.h5':
        raise ValueError('Weights filepath must have `.h5` extension.')

    with open(arch_fpath, "r") as yaml_file:
        yaml_architecture = yaml_file.read()

    # Required since we're using a custom layer within Keras
    # custom_objects = {'BilinearUpsampling': BilinearUpsampling}

    if n_gpus > 1:
        # Load weights on CPU to avoid taking up GPU space
        with tf.device('/cpu:0'):
            template_model = model_from_yaml(yaml_architecture)
            template_model.load_weights(weights_fpath)

            for layer in template_model.layers:
                layer.trainable = False

        model = multi_gpu_model(template_model, gpus=n_gpus)
    # If on only 1 gpu (or cpu), train as normal
    else:
        model = model_from_yaml(yaml_architecture)
        model.load_weights(weights_fpath)

        for layer in model.layers:
            layer.trainable = False

    return model

model_weight = '/Users/yizhuangfang/Documents/Development_Seed/project-connect/phase2/main_model/models/1218_131408_L0.54_E05_weights.h5'
model_arc = '/Users/yizhuangfang/Documents/Development_Seed/project-connect/phase2/main_model/models/1218_131408_arch.yaml'
model_params = '/Users/yizhuangfang/Documents/Development_Seed/project-connect/phase2/main_model/models/1218_131408_params.yaml'
export_path = '/Users/yizhuangfang/Documents/Development_Seed/project-connect/phase2/main_model/sat_xception_tf-serving/1'

model = load_keras_model(model_arc,model_weight)

model_builder = builder.SavedModelBuilder(export_path)
signature = predict_signature_def(inputs={'input': model.inputs[0]},
                                  outputs={'output': model.outputs[0]})

# Get a Keras session and set the signature for prediction
with K.get_session() as sess:
    model_builder.add_meta_graph_and_variables(
        sess=sess, tags=[tag_constants.SERVING], clear_devices=True,
        signature_def_map={signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature})

    model_builder.save()
