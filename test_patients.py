import multiprocessing
import os, warnings
import time

import numpy as np
import torch
import utils.exp_utils as utils
from evaluator import Evaluator
from predictor_patients import Predictor,apply_wbc_to_patient,merge_2D_to_3D_preds_per_patient
from plotting import plot_batch_prediction
from tqdm import tqdm
import argparse

for msg in ["Attempting to set identical bottom==top results",
            "This figure includes Axes that are not compatible with tight_layout",
            "Data has no positive values, and therefore cannot be log-scaled.",
            ".*invalid value encountered in double_scalars.*",
            ".*Mean of empty slice.*"]:
    warnings.filterwarnings("ignore", msg)
import shutil
import pandas as pd
import pickle
from multiprocessing import Pool

def test_singal_patient(patient_pid):
    patient_pid=patient_pid[1] #(index,pid)
    # final_result is exist
    # final_result="../rifrac_test/fold_0/final_pred_boxes_hold_out_list.pickle"
    final_result = None
    with torch.cuda.device('cuda:0'):
        # logger.info('starting testing model of in exp {}'.format(cf.exp_dir))
        net = model.net(cf, logger).cuda()
        test_predictor = Predictor(cf, net, logger, mode='test')
        test_evaluator = Evaluator(cf, logger, mode='test')
        if final_result is None:
            batch_gen = data_loader.get_test_generator_test(cf, logger,data_set)
            batch_gen=batch_gen['test']
            test_results_list = test_predictor.predict_test_set_all(batch_gen,patient_pid, result_dir,return_results=False)
        else:
            test_results_list = pd.read_pickle(final_result)
        print("have finshed [{}] patient".format(patient_pid),flush=True)

def test_patients(data_set,test_ix=None,process_num=1):
    fold=cf.fold
    if test_ix is None:
        process_num=4
        with open(os.path.join(exp_dir, 'fold_ids.pickle'), 'rb') as handle:
            fold_list = pickle.load(handle)
        if data_set=='val':
            _, test_ix,_, _ = fold_list[fold]
            # test_ix=test_ix[:15]
        elif data_set=='train':
            train_ix,_,test_ix, _ = fold_list[fold]
            test_ix=np.concatenate((train_ix,test_ix))
        test_ix=range(len(test_ix)+1)
    print("Testing [{}] mode in [{}_set] all [{}] patients by [{}] processes".format(function_apply,data_set,len(test_ix),process_num))
    pool = Pool(processes=process_num)
    p1 = pool.map(test_singal_patient, enumerate(test_ix), chunksize=1)
    pool.close()
    pool.join()

# query patients which have not be tested
def query_unusual(data_set):
    fold_id_path=os.path.join(exp_dir,"fold_ids.pickle")
    fold_list = pd.read_pickle(fold_id_path)
    if data_set=='train':
        train_ixs, _, test_ixs, _ = fold_list[0]
        subset_ixs=np.concatenate((train_ixs,test_ixs))
    elif data_set=='val':
        _, subset_ixs, _, _ = fold_list[0]

    p_df = pd.read_pickle(os.path.join(cf.pp_data_path, cf.input_df_name))
    subset_pids = [np.unique(p_df.pid.tolist())[ix] for ix in subset_ixs]
    p_df = p_df[p_df.pid.isin(subset_pids)]
    pids = p_df.pid.tolist()

    # finished result
    path=os.path.join(cf.fold_dir,'test')
    results = [i.split('_')[0] for i in os.listdir(path)]
    index = []
    for i, data in enumerate(pids):
        if data not in results:
            index.append(i)
    print(index)
    test_patients(data_set,test_ix=index)

def get_postprocess_result(results_per_patient):
    n_ens = 1
    # all_final_result = all_final_result + results_per_patient
    final_patient_box_results = [(res_dict["boxes"], pid) for res_dict, pid in results_per_patient]
    # # consolidate predictions.
    # logger.info('applying wcs to test set predictions with iou = {} and n_ens = {}.'.format(
    #     cf.wcs_iou, n_ens))
    pool = Pool(processes=6)
    mp_inputs = [[ii[0], ii[1], cf.class_dict, cf.wcs_iou, n_ens] for ii in
                 final_patient_box_results]
    final_patient_box_results = pool.map(apply_wbc_to_patient, mp_inputs, chunksize=1)
    pool.close()
    pool.join()

    # merge 2D boxes to 3D cubes. (if model predicts 2D but evaluation is run in 3D)
    if cf.merge_2D_to_3D_preds:
        logger.info(
            'applying 2Dto3D merging to test set predictions with iou = {}.'.format(cf.merge_3D_iou))
        pool = Pool(processes=6)
        mp_inputs = [[ii[0], ii[1], cf.class_dict, cf.merge_3D_iou] for ii in final_patient_box_results]
        final_patient_box_results = pool.map(merge_2D_to_3D_preds_per_patient, mp_inputs, chunksize=1)
        pool.close()
        pool.join()

    # final_patient_box_results holds [avg_boxes, pid] if wbc
    for ix in range(len(results_per_patient)):
        assert results_per_patient[ix][1] == final_patient_box_results[ix][1], "should be same pid"
        results_per_patient[ix][0]["boxes"] = final_patient_box_results[ix][0]
    return results_per_patient

# test_df
def get_singal_result(raw_file,test_evaluator):
    results_per_patient = pd.read_pickle(raw_file)
    test_evaluator.evaluate_predictions(results_per_patient)
    test_df=test_evaluator.test_df
    base_path=os.path.join(result_dir,"test_df_csv")
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    save_path=os.path.join(base_path,"raw_test_df"+str(test_df.pid[0])+'.csv')
    test_df.to_csv(save_path)

    #wbc
    afterWbc_results_per_patient=get_postprocess_result(results_per_patient)
    test_evaluator.evaluate_predictions(afterWbc_results_per_patient)
    wbc_test_df=test_evaluator.test_df
    save_path=os.path.join(base_path,"final_test_df"+str(wbc_test_df.pid[0])+'.csv')
    wbc_test_df.to_csv(save_path)

    return test_df,wbc_test_df



def get_result(result_dir):
    result_dir=os.path.join(result_dir,'test')
    raw_result_list=[i for i in os.listdir(result_dir) if "raw_pred" in i]
    # raw_result_list=raw_result_list[:2]

    test_evaluator = Evaluator(cf, logger, mode='test')
    all_test_df=[]
    wbc_all_test_df=[]
    for i in tqdm(raw_result_list):
        signal_result = os.path.join(result_dir, i)
        test_df,wbc_test_df=get_singal_result(signal_result,test_evaluator)
        all_test_df.append(test_df)
        wbc_all_test_df.append(wbc_test_df)

    all_test_df=pd.concat(all_test_df)
    wbc_all_test_df=pd.concat(wbc_all_test_df)

    # save the result
    dst_dir=os.path.split(result_dir)[:-1][0]
    scr_dir=os.path.join(exp_dir,"test","test")
    raw_dst_dir=os.path.join(dst_dir,'raw')
    wbc_dst_dir = os.path.join(dst_dir, 'wbc')
    if os.path.exists(raw_dst_dir):
        shutil.rmtree(raw_dst_dir)
    if os.path.exists(wbc_dst_dir):
        shutil.rmtree(wbc_dst_dir)

    test_evaluator.test_df=all_test_df
    test_evaluator.score_test_df()
    shutil.move(scr_dir,raw_dst_dir)

    # wbc
    test_evaluator.test_df = wbc_all_test_df
    test_evaluator.score_test_df()
    shutil.move(scr_dir, wbc_dst_dir)

    save_all_test_df = os.path.join(raw_dst_dir,"all_raw_test_df.csv")
    save_wbc_all_test_df = os.path.join(wbc_dst_dir,"all_final_test_df.csv")
    all_test_df.to_csv(save_all_test_df)
    wbc_all_test_df.to_csv(save_wbc_all_test_df)


parser = argparse.ArgumentParser()
parser.add_argument('-f', '--fun', type=str,  default='get',
                    help='function_apply: get / query / res')
parser.add_argument('-d', '--data_set', type=str,  default='train',
                    help='test data_set: train / val')
parser.add_argument('-result_dir', type=str, default="/media/victoria/9c3e912e-22e1-476a-ad55-181dbde9d785/jinxiaoqiang/rifrac/experiment/RetinaNet_3D",
                        help='path to test dir. will be created if non existent.')
parser.add_argument('-exp_dir', type=str, default='experiments/rifrac_RetinaNet_debug',
                        help='path to experiment dir.')
parser.add_argument('-exp_source', type=str, default='experiments/rifrac_RetinaNet_exp',
                    help='specifies, from which source experiment to load configs and data_loader.')
args = parser.parse_args()
"""
need to change
"""
# which dataset to test
function_apply=args.fun # get: get raw; query: get unnormal; res: get metrics
data_set=args.data_set
result_dir=args.result_dir
result_dir=os.path.join(result_dir,data_set)
if not os.path.exists(result_dir):
    os.makedirs(result_dir)
exp_source = args.exp_source
exp_dir = args.exp_dir

"""
not change
"""
server_env = False
logger = utils.get_logger(result_dir, server_env)
cf = utils.prep_exp(exp_source, exp_dir, server_env, is_training=False)
cf.hold_out_test_set = False
cf.test_aug = False
cf.pp_test_name = cf.pp_name  # not useful
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
cf.save_batch_data = False
# if save final result
cf.save_final_result = True
data_loader = utils.import_module('dl', os.path.join(exp_source, 'data_loader.py'))
model = utils.import_module('model', cf.model_path)
logger.info("loaded model from {}".format(cf.model_path))
cf.fold = 0
cf.fold_dir=os.path.join(exp_dir, 'fold_0')
if __name__=='__main__':
    print("begin to run [{}] mode".format(function_apply))
    begin_test = time.time()
    if function_apply=='get':
        test_patients(data_set)
    elif function_apply=='query':
        query_unusual(data_set)
    elif function_apply=='res':
        get_result(result_dir)
    else:
        raise Exception("please choose function")
    end_test=time.time()
    print("[{}] spend time={}".format(function_apply,end_test - begin_test))