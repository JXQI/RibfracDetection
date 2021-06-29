import pandas as pd
import os
import numpy as np
import SimpleITK as sitk
import pickle
from os.path import join
from tqdm import tqdm

"""
    function:生成seg的npz文件和seg_result.csv (include confidence)
    args:
    return:
"""
def product_seg(dstpath,result,scores,csvpath="./csv"):
    boxes = result[0]['boxes'][0]
    seg_preds = result[0]["seg_preds"][0][0]  # (1,1,y,x,z)
    seg_preds = np.transpose(seg_preds, axes=(2, 0, 1))

    pid = result[1]
    # print("boxes's keys={}".format(boxes[0].keys()))
    gt_bbox_list = []  # gt bbox (y1,y2,x1,x2,z1,z2)
    det_bbox_list = []  # test result bbox

    for i in range(len(boxes)):
        if (boxes[i]["box_type"] == 'gt'):
            gt_bbox_list.append(boxes[i])
        elif (boxes[i]["box_type"] == 'det' and boxes[i]["box_score"] > scores):
            det_bbox_list.append(boxes[i])
        else:
            pass
    print("box_num={},gt_num={},det_num={}".format(len(boxes), len(gt_bbox_list), len(det_bbox_list)))

    #  boxes of detect result
    seg_preds[seg_preds >0] = 1 #TODO:暂时将边界全部不当作seg结果
    seg_det = np.zeros(seg_preds.shape)
    label_id=1 # 标注检测到的骨折部分mask
    seg_results={"public_id":[],"label_id":[],"confidence":[],"label_code":[]}
    for value in det_bbox_list:
        item = np.array(value["box_coords"], dtype=int).tolist()
        confidence=value["box_score"]
        if np.sum(seg_preds[item[-2]:item[-1], item[0]:item[2], item[1]:item[3]]):
            seg_det[item[-2]:item[-1], item[0]:item[2], item[1]:item[3]] = label_id
            seg_results["public_id"].append(pid)
            seg_results["label_id"].append(label_id)
            seg_results["confidence"].append(confidence)
            seg_results["label_code"].append(-1)
            label_id+=1
    seg_preds = seg_preds * seg_det
    seg_preds = seg_preds.astype("uint8")

    # np.savez_compressed(join(dstpath,det_seg_name),seg_preds)
    df=pd.DataFrame.from_dict(seg_results)
    df=df.sort_values(by='confidence',ascending=False)
    # df.to_csv(join(dstpath,pid+'-pred.csv'))
    df.to_csv(join(csvpath, '{}-pred.csv'.format(pid)),index=False)

    # val result
    det_seg_name = pid + "-label.nii.gz"
    seg_preds_nii = sitk.GetImageFromArray(seg_preds)
    sitk.WriteImage(seg_preds_nii, os.path.join(dstpath, "{}".format(det_seg_name)))

def agg_csv(src_path,dst_path):
    pred_csv =pd.DataFrame.from_dict({"public_id": [], "label_id": [], "confidence": [], "label_code": []})
    for i in os.listdir(src_path):
        if i.endswith(".csv"):
            sigle_csv=pd.read_csv(join(src_path,i))
            pred_csv=pred_csv.append(sigle_csv,ignore_index=True)
    pred_csv.to_csv(join(dst_path,'ribfrc-train-pred.csv'),index=False)

"""
function: 生成对应的label
args: detresult_path 检测的结果
    npz_path 标注的路径
    label_file 标注列表
    label_csvpath 单个病人的标注列表
    labelpath 保存的位置
"""
def get_mul_label(detresult_path,npz_path,label_file,label_csvpath,labelpath):
    pid_list=[item.split("_")[0] for item in os.listdir(detresult_path)]
    for pid in tqdm(pid_list):
        npz_file = join(npz_path, "{}_img.npz".format(pid))
        label = np.load(npz_file)['rois']
        label_nii = sitk.GetImageFromArray(label)
        sitk.WriteImage(label_nii, join(labelpath, "{}-label.nii.gz".format(pid)))

        label_csv = pd.read_csv(label_file)
        label_csv = label_csv[label_csv.public_id == pid]
        label_csv.to_csv(join(label_csvpath, "{}-train-info.csv".format(pid)), index=False)


"""
function:批量生成seg
args: detresult_path 检测结果所在路径
    dstpath 生成结果的保存位置
    pred_csvpath 每个病人预测的结果保存位置
"""
def get_mul_seg(detresult_path,dstpath,pred_csvpath):
    for i in tqdm(os.listdir(detresult_path)):
        if i.endswith("pickle"):
            result = pd.read_pickle(join(detresult_path,i))[0]
            product_seg(dstpath, result, scores=0.1, csvpath=pred_csvpath)

"""
function: 生成结果
"""
from ribfrac import evaluation
def seg_result(detresult_path,dstpath,pred_csvpath,\
               npz_path,label_file,label_csvpath,labelpath):
    get_mul_seg(detresult_path, dstpath, pred_csvpath)
    get_mul_label(detresult_path, npz_path, label_file, label_csvpath,labelpath)
    agg_csv(pred_csvpath, dstpath)
    agg_csv(label_csvpath, labelpath)

    # 评估结果
    eval_results=evaluation.evaluate(labelpath,dstpath)
    # eval_results = evaluation.evaluate("label", "pred_seg_final")
    df=pd.DataFrame.from_dict(eval_results)
    df.to_csv(join(pred_csvpath,"result.csv"),index=False)


if __name__=="__main__":
    basepath="/media/victoria/9c3e912e-22e1-476a-ad55-181dbde9d785/jinxiaoqiang/rifrac/experiment/RetinaNet_3D_segment_boxseg_pretrain/val"
    detresult_path=join(basepath,"test")
    dstpath=join(basepath,"seg")
    pred_csvpath=join(basepath,"pred_csvs")
    npz_path="/home/victoria/train_data_jinxiaoqiang/ribfrac_train/data_segment_npz"
    label_file="/home/victoria/train_data_jinxiaoqiang/ribfrac_train/ribfrac-train-info.csv"
    label_csvpath=join(basepath,"label_csvs")
    labelpath=join(basepath,"seg_label")

    if not os.path.isdir(dstpath):
        os.makedirs(dstpath)
    if not os.path.isdir(pred_csvpath):
        os.makedirs(pred_csvpath)
    if not os.path.isdir(label_csvpath):
        os.makedirs(label_csvpath)
    if not os.path.isdir(labelpath):
        os.makedirs(labelpath)

    seg_result(detresult_path, dstpath, pred_csvpath,
               npz_path, label_file, label_csvpath, labelpath)
    # 测试单个结果
    # pid = "1"
    # pred_csvpath="./pred_csv"
    # label_csvpath="./label_csv"
    # if not os.path.isdir(pred_csvpath):
    #     os.makedirs(pred_csvpath)
    # if not os.path.isdir(label_csvpath):
    #     os.makedirs(label_csvpath)
    #
    #
    # dstpath="./pred_seg_final"
    # if not os.path.isdir(dstpath):
    #     os.makedirs(dstpath)
    # result_path = "../experiments/rifrac_RetinaNet_segment_boxseg_exp/examples/{}/final_pred_boxes_hold_out_list.pickle".format(pid)
    # result = pd.read_pickle(result_path)[0]
    # product_seg(dstpath, result, 0.1, gd_box=False, csvpath=pred_csvpath)
    #
    # # label
    # labelpath="./label"
    # if not os.path.isdir(labelpath):
    #     os.makedirs(labelpath)
    # path="/home/victoria/train_data_jinxiaoqiang/ribfrac_train/data_segment_npz"
    # npz_file=join(path,"RibFrac{}_img.npz".format(pid))
    # label=np.load(npz_file)['rois']
    # label_nii=sitk.GetImageFromArray(label)
    # sitk.WriteImage(label_nii,join(labelpath,"RibFrac{}-label.nii.gz".format(pid)))
    # label_csv_path="/home/victoria/train_data_jinxiaoqiang/ribfrac_train/ribfrac-train-info.csv"
    # label_csv=pd.read_csv(label_csv_path)
    # label_csv=label_csv[label_csv.public_id=="RibFrac{}".format(pid)]
    # label_csv.to_csv(join(label_csvpath,"RibFrac{}-train-info.csv".format(pid)),index=False)
    #
    # agg_csv(pred_csvpath,dstpath)
    # agg_csv(label_csvpath,labelpath)
