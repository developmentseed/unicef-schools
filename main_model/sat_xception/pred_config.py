
"""
pred_config.py

List some configuration parameters for prediction
"""
import os
from os import makedirs, path as op

preds_dir = op.join(os.getcwd(), "preds")
plot_dir = op.join(os.getcwd(), "plots")
ckpt_dir = op.join(os.getcwd(), "models")

# s3://project-connect-nana-share/phase_2/
# Params for downloading tiles from Mapbox to S3 for later prediction
download_params = dict(aws_bucket_name='project-connect-nana-share',
                       aws_dir='phase_2/main_model/',
                       aws_region='us-east-1',
                       tile_ind_list=op.join(preds_dir, 'clombia_test.txt'),
                       tile_ind_list_format=['cover', 'tabbed', 'spaced'][2],  # cover.js or tab/space separated
                       n_green_threads=500,
                       download_prob=0.05,  # 0.1 downloads 10%, 0.9 downloads 90%
                       url_template='https://api.mapbox.com/v4/digitalglobe.2lnpeioh/{z}/{x}/{y}.png?access_token={token}'.format(
                           x='{x}', y='{y}', z='{z}', token="pk.eyJ1IjoiZGlnaXRhbGdsb2JlIiwiYSI6ImNpbWdrZjhlZjAwMnd0emtvNXYzYmFwZm4ifQ.5owPkJs5HMvoB8IQPyuwEw"))

# Params to get a list of tile indices from a geojson boundary
gen_tile_inds_params = dict(geojson_bounds=op.join(preds_dir, 'colombia_test.geojson'),
                            geojson_pakistan_bounds=op.join(preds_dir, 'colombia_test.geojson'),  # Seperate bounding box for Colombia
                            country='Colombia',
                            max_zoom=17)

# Params to run inference on some tiles in S3
pred_params = dict(aws_bucket_name='project-connect-nana-share',
                   pred_fname='colombia_test.json',  # File name for predictions
                   aws_pred_dir='phase_2/main_model/colombia_test/',  # File dir for prediction values
                   local_img_dir=op.join(preds_dir, 'colombia_test'),
                   model_time='1218_131408',
                   single_batch_size=16,  # Number of images seen by a single GPU
                   n_gpus=1,
                   deci_prec=4)  # Number of decimal places in prediction precision
pred_params.update(dict(model_arch_fname='{}_arch.yaml'.format(pred_params['model_time']),
                        model_params_fname='{}_params.yaml'.format(pred_params['model_time']),
                        model_weights_fname='{}_L0.54_E05_weights.h5'.format(pred_params['model_time'])))

pred_fnames = ['colombia_test.json']

geojson_out_fnames = ['colombia_test.geojson']

gen_geojson_params = dict(upper_thresh_lims=[0.92, 1.],
                          #upper_thresh_lims=[0.95, 0.98, 1.],
                          thresh_labels=['Maybe', 'Yes'],
                          #thresh_labels=['No', 'Maybe', 'Yes'],
                          thresh_cols=['#888888', '#ffff00'],
                          #thresh_cols=['#888888', ' #ff8000', '#ffff00'],
                          exclude_subthresh=True,
                          pred_fnames=pred_fnames,
                          geojson_out_fnames=geojson_out_fnames,
                          deci_prec=4)
