import os
import sys
import argparse
import logging
from os import makedirs, path as op


from sat_xception.version import __version__
from sat_xception.train import train
from sat_xception.predict import predict

logger = logging.getLogger(__name__)

def parse_args(args):
    desc = 'pixel_decoder (v%s)' % __version__
    dhf = argparse.ArgumentDefaultsHelpFormatter
    parser0 = argparse.ArgumentParser(description=desc)

    pparser = argparse.ArgumentParser(add_help=False)
    pparser.add_argument('--version', help='Print version and exit', action='version', version=__version__)
    pparser.add_argument('--log', default=2, type=int,
                         help='0:all, 1:debug, 2:info, 3:warning, 4:error, 5:critical')

    subparsers = parser0.add_subparsers(dest='command')

    parser = subparsers.add_parser('train', parents=[pparser], help='train the model', formatter_class=dhf)
    parser.add_argument('-model', '--model_id', help='pre-trained model to fine tune image classification', default="xception", type=str, required=True)
    parser.add_argument('-train', '--train_dir', help='path to training categories', default="train", type=str, required=True)
    parser.add_argument('-valid', '--validation_dir', help='path to validation dataset', default="test", type=str, required=True)

    #
    parser = subparsers.add_parser('predict', parents=[pparser], help='predict with test data', formatter_class=dhf)
    # parser.add_argument('-pred', '--prediction_folder', help='path to test tiles for running prediction', type=str, required=True)
    parsed_args = vars(parser0.parse_args(args))

    return parsed_args

def main(cmd, **kwargs):
    if cmd == 'train':
        train(**kwargs)
    elif cmd == 'predict':
        predict(**kwargs)
# def main(cmd, batch_size, imgs_folder, test_folder, masks_folder, models_folder,pred_folder, model_id, origin_shape_no, border_no, channel_no):
#     if cmd == 'train':
#         train(batch_size, imgs_folder, masks_folder, models_folder, model_id, origin_shape_no, border_no, channel_no)
#     elif cmd == 'predict':
#         predict(imgs_folder, test_folder, models_folder, pred_folder, origin_shape_no, border_no, channel_no, model_id)


def cli():
    args = parse_args(sys.argv[1:])
    logger.setLevel(args.pop('log') * 10)
    # cmd = args.pop('command')
    # out_folder = args.get('trained_model')
    #
    # # create destination directory to save the trained model
    # if not op.isdir(op.join(os.getcwd(),out_folder)):
    #     makedirs(out_folder)
    main(args.pop('command'), **args)


if __name__ == "__main__":
    cli()
