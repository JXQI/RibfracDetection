import argparse
import os, warnings
import time

import torch

import utils.exp_utils as utils
from evaluator import Evaluator
from predictor import Predictor
from plotting import plot_batch_prediction

for msg in ["Attempting to set identical bottom==top results",
            "This figure includes Axes that are not compatible with tight_layout",
            "Data has no positive values, and therefore cannot be log-scaled.",
            ".*invalid value encountered in double_scalars.*",
            ".*Mean of empty slice.*"]:
    warnings.filterwarnings("ignore", msg)
import shutil

def test(logger):
    """
    perform testing for a given fold (or hold out set). save stats in evaluator.
    """
    logger.info('starting testing model of in exp {}'.format(cf.exp_dir))
    net = model.net(cf, logger).cuda()
    test_predictor = Predictor(cf, net, logger, mode='test')
    test_evaluator = Evaluator(cf, logger, mode='test')
    batch_gen = data_loader.get_test_generator(cf, logger)
    test_results_list = test_predictor.predict_test_set(batch_gen, return_results=True)
    test_evaluator.evaluate_predictions(test_results_list)
    test_evaluator.score_test_df()

if __name__ == '__main__':
    stime = time.time()
    exp_source="/Users/jinxiaoqiang/jinxiaoqiang/medicaldetectiontoolkit/experiments/rifrac_exp"
    exp_dir="/Users/jinxiaoqiang/jinxiaoqiang/medicaldetectiontoolkit/experiments/rifrac_exp"
    server_env=False
    cf = utils.prep_exp(exp_source, exp_dir, server_env, is_training=False, use_stored_settings=True)

    # create the new testset
    file_pid = "421"
    test_path = "./examples"
    if not os.path.isdir(test_path):
        os.makedirs(test_path)
    examples_image = os.path.join(cf.pp_data_path,"RibFrac{}_img.npy".format(file_pid))
    examples_label = os.path.join(cf.pp_data_path, "RibFrac{}_rois.npy".format(file_pid))
    examples_info  = os.path.join(cf.pp_data_path, "meta_info_RibFrac{}.pickle".format(file_pid))
    shutil.copy(examples_image,os.path.join(test_path,"RibFrac{}_img.npy".format(file_pid)))
    shutil.copy(examples_label, os.path.join(test_path, "RibFrac{}_rois.npy".format(file_pid)))
    shutil.copy(examples_info, os.path.join(test_path, cf.input_df_name))

    cf.pp_test_data_path=test_path
    cf.hold_out_test_set = True
    cf.test_aug=False


    if args.mode == 'test':
        cf.data_dest = args.data_dest
        logger = utils.get_logger(cf.exp_dir, cf.server_env)
        data_loader = utils.import_module('dl', os.path.join(args.exp_source, 'data_loader.py'))
        model = utils.import_module('model', cf.model_path)

        logger.info("loaded model from {}".format(cf.model_path))

        with torch.cuda.device(args.cuda_device):
            test(logger)