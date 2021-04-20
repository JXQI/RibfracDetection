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
import pandas as pd
import pickle

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
    exp_source="../rifrac_exp/"
    exp_dir="../rifrac_test/"
    server_env=False
    cf = utils.prep_exp(exp_source, exp_dir, server_env, is_training=False, use_stored_settings=True)

    # create the new testset
    file_pid = "500"
    test_path = "./examples"
    if not os.path.isdir(test_path):
        os.makedirs(test_path)
    examples_image = os.path.join(cf.pp_data_path,"RibFrac{}_img.npy".format(file_pid))
    examples_label = os.path.join(cf.pp_data_path, "RibFrac{}_rois.npy".format(file_pid))
    examples_info  = os.path.join(cf.pp_data_path, "meta_info_RibFrac{}.pickle".format(file_pid))
    shutil.copy(examples_image,os.path.join(test_path,"RibFrac{}_img.npy".format(file_pid)))
    shutil.copy(examples_label, os.path.join(test_path, "RibFrac{}_rois.npy".format(file_pid)))
    shutil.copy(examples_info, os.path.join(test_path, "meta_info_RibFrac{}.pickle".format(file_pid)))
    df = pd.DataFrame(columns=['pid', 'class_target', 'spacing', 'fg_slices'])
    info_file=os.path.join(test_path, "meta_info_RibFrac{}.pickle".format(file_pid))
    with open(info_file, 'rb') as handle:
        df.loc[len(df)] = pickle.load(handle)
    df.to_pickle(os.path.join(test_path, cf.input_df_name))

    cf.pp_test_data_path=test_path
    cf.hold_out_test_set = True
    cf.test_aug=False
    cf.fold_dir=os.path.join(exp_dir,'fold_0')
    cf.fold='fold_0'
    cf.pp_test_name=cf.pp_name #not useful
    # select a maximum number of patient cases to test. number or "all" for all
    cf.max_test_patients = 1
    # set the top-n-epochs to be saved for temporal averaging in testing.
    cf.save_n_models = 1
    cf.test_n_epochs = 1

    logger = utils.get_logger(exp_dir, server_env)
    data_loader = utils.import_module('dl', os.path.join(exp_source, 'data_loader.py'))
    model = utils.import_module('model', cf.model_path)

    logger.info("loaded model from {}".format(cf.model_path))

    with torch.cuda.device('cuda:0'):
        test(logger)