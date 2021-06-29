import pandas as pd
import os
import numpy as np
import SimpleITK as sitk
import pickle
from os.path import join

#TODO: 1.afffine 2.csv

def get_seg_from_box(dstpath,result,scores, gd_box=False):
    boxes = result[0]['boxes'][0]
    seg_preds = result[0]["seg_preds"][0][0]  # (1,1,y,x,z)
    seg_preds = np.transpose(seg_preds, axes=(2, 0, 1))

    pid = result[1]
    print("boxes's keys={}".format(boxes[0].keys()))
    gt_bbox_list = []  # gt bbox (y1,y2,x1,x2,z1,z2)
    det_bbox_list = []  # test result bbox

    for i in range(len(boxes)):
        if (boxes[i]["box_type"] == 'gt'):
            gt_bbox_list.append(boxes[i])
        elif (boxes[i]["box_type"] == 'det' and boxes[i]["box_score"] > scores):
            det_bbox_list.append(boxes[i])
        else:
            pass
            # print(boxes[i]["box_type"])
    print("box_num={},gt_num={},det_num={}".format(len(boxes), len(gt_bbox_list), len(det_bbox_list)))
    # to nii.gz
    det_seg_name = "segDet_raw.nii.gz" if gd_box else "segDet_final.nii.gz"

    #  boxes of detect result
    # print("det_bbox example:\n{}".format(det_bbox_list[-1]))
    seg_det = np.zeros(seg_preds.shape)
    for value in det_bbox_list:
        item = np.array(value["box_coords"], dtype=int).tolist()
        seg_det[item[-2]:item[-1], item[0]:item[2], item[1]:item[3]] = 1
    seg_preds = seg_preds * seg_det
    seg_preds[seg_preds>0]=1
    seg_preds=seg_preds.astype("uint8")

    seg_preds_nii = sitk.GetImageFromArray(seg_preds)
    sitk.WriteImage(seg_preds_nii, os.path.join(dstpath, "{}_{}".format(pid, det_seg_name)))


if __name__=="__main__":
    result_path="../experiments/rifrac_RetinaNet_segment_seg_exp/examples/1/final_pred_boxes_hold_out_list.pickle"
    dst_path="./debug_dir"
    if not os.path.isdir(dst_path):
        os.makedirs(dst_path)
    result=pd.read_pickle(result_path)[0]
    get_seg_from_box(dst_path,result,0.1)