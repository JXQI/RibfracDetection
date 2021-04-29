import argparse
import os, warnings
import time
import torch
import utils.exp_utils as utils
from evaluator import Evaluator
from predictor import Predictor
from plotting import plot_batch_prediction
from show_npy import test_result

for msg in ["Attempting to set identical bottom==top results",
            "This figure includes Axes that are not compatible with tight_layout",
            "Data has no positive values, and therefore cannot be log-scaled.",
            ".*invalid value encountered in double_scalars.*",
            ".*Mean of empty slice.*"]:
    warnings.filterwarnings("ignore", msg)
import shutil
import pandas as pd
import pickle

def test(logger,result_exist=None):
    """
    perform testing for a given fold (or hold out set). save stats in evaluator.
    """
    logger.info('starting testing model of in exp {}'.format(cf.exp_dir))
    net = model.net(cf, logger).cuda()
    test_predictor = Predictor(cf, net, logger, mode='test')
    test_evaluator = Evaluator(cf, logger, mode='test')
    if result_exist is None:
        batch_gen = data_loader.get_test_generator(cf, logger)
        test_results_list = test_predictor.predict_test_set(batch_gen, return_results=True)
    else:
        test_results_list=pd.read_pickle(result_exist)
    test_evaluator.evaluate_predictions(test_results_list)
    test_evaluator.score_test_df()

# test pid
file_pid = "498"
if __name__ == '__main__':
    stime = time.time()
    exp_source="../rifrac_exp/"
    exp_dir="../rifrac_test/"
    server_env=False
    cf = utils.prep_exp(exp_source, exp_dir, server_env, is_training=False, use_stored_settings=True)

    # create the new testset
    # file_pid = "499"
    test_path = os.path.join("./examples",file_pid)
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
    # disable the re-sampling of mask proposals to original size for speed-up.
    # since evaluation is detection-driven (box-matching) and not instance segmentation-driven (iou-matching),
    # mask-outputs are optional.
    cf.return_masks_in_test = True
    # if save batch data
    cf.save_batch_data=True
    # if save final result
    cf.save_final_result=True

    logger = utils.get_logger(exp_dir, server_env)
    data_loader = utils.import_module('dl', os.path.join(exp_source, 'data_loader.py'))
    model = utils.import_module('model', cf.model_path)

    logger.info("loaded model from {}".format(cf.model_path))

    # final_result is exist
    # final_result="../rifrac_test/fold_0/final_pred_boxes_hold_out_list.pickle"
    final_result=None
    with torch.cuda.device('cuda:0'):
        test(logger,final_result)

    # mv the result file to dst
    raw_result=os.path.join(cf.fold_dir,"raw_pred_boxes_hold_out_list.pickle")
    final_result=os.path.join(cf.fold_dir,"final_pred_boxes_hold_out_list.pickle")

    batch_pickle=os.path.join(cf.fold_dir,"RibFrac{}batch.pickle".format(file_pid))

    test_metric_pickle=os.path.join(exp_dir,'test',"fold_0_test_df.pickle")
    test_metric_txt=os.path.join(exp_dir,'test',"results.txt")

    shutil.copy(raw_result, test_path)
    shutil.copy(final_result,test_path)
    shutil.copy(test_metric_pickle,test_path)
    shutil.copy(test_metric_txt,test_path)
    shutil.copy(batch_pickle,test_path)

    # calcuate result
    result = test_result(test_path)
    # product nii.gz
    result.read_batchData_pickle()
    # return recall and precision
    result.deal_metrics()