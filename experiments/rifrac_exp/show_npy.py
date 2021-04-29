import pandas as pd
import os
import numpy as np
import SimpleITK as sitk
import pickle
from os.path import join
import configs
cf = configs.configs()

# show npy file, change .npy file to nii.gz ,val the correct of data
def show_npy(root_dir,pid):
    meta_info=os.path.join(root_dir,"meta_info_RibFrac{}.pickle".format(pid))
    image=os.path.join(root_dir,"RibFrac{}_img.npy".format(pid))
    label=os.path.join(root_dir,"RibFrac{}_rois.npy".format(pid))
    image_nii_gz=os.path.join(root_dir,"RibFrac{}_image.nii.gz".format(pid))
    label_nii_gz=os.path.join(root_dir,"RibFrac{}_label.nii.gz".format(pid))

    # read pid list
    info=pd.read_pickle(meta_info)
    print(info)

    # show the whole image.npy/label.npy
    image_nii=sitk.GetImageFromArray(np.load(image))
    label_nii=sitk.GetImageFromArray(np.load(label))
    sitk.WriteImage(image_nii,image_nii_gz)
    sitk.WriteImage(label_nii,label_nii_gz)
    print("end!")

# read the fold_ids.pickle about the split of train/val/test
def read_fold_ids(path):
    info=pd.read_pickle(path)
    print(info.columns)
    print(info["pid"])
    print(info["class_target"])

# show the 3D batch-data
def show_batchdata(root_dir,pid):
    image = os.path.join(root_dir, "RibFrac{}_data.npy".format(pid))
    label = os.path.join(root_dir, "RibFrac{}_label.npy".format(pid))
    image_nii_gz = os.path.join(root_dir, "RibFrac{}_data.nii.gz".format(pid))
    label_nii_gz = os.path.join(root_dir, "RibFrac{}_label.nii.gz".format(pid))

    # show the whole image.npy/label.npy
    image_nii = sitk.GetImageFromArray(np.load(image))
    label_nii = sitk.GetImageFromArray(np.load(label))
    sitk.WriteImage(image_nii, image_nii_gz)
    sitk.WriteImage(label_nii, label_nii_gz)
    print("end!")

"""
    function:
        produce a seg nii.gz file
    args:
        path: raw_pred_boxes_hold_out_list.pickle
"""
def load_seg_from_result(path):
    df=pd.read_pickle(path)
    print(len(df[0][0]['boxes']))
    print(df[0][0]['boxes'][0][-1])
    seg=np.array(df[0][0]['seg_preds'][0][0],dtype=np.int8)
    print(seg.shape)
    # (z,x,y)
    seg=np.transpose(seg,axes=(2,1,0))
    seg_nii=sitk.GetImageFromArray(seg)
    path=path.split('/')
    seg_nii_name=join(path[0],path[1],path[2],"RibFrac{}_seg.nii.gz".format(path[2]))
    print(seg_nii_name)
    sitk.WriteImage(seg_nii,seg_nii_name)


def Dict2df(path):
    df = pd.DataFrame(columns=['pid', 'class_target', 'spacing', 'fg_slices'])
    with open(path,'rb') as handle:
        df.loc[len(df)] = pickle.load(handle)
    print(df)
def dpy2niigz(path,dstpath):
    image_nii = sitk.GetImageFromArray(np.load(path))
    image_nii_gz = os.path.join(dstpath,path.split('/')[-1].split('.')[0]+'.nii.gz')
    sitk.WriteImage(image_nii, image_nii_gz)
    print("end!")

"""
    function: read a batch_data pickle
    return: index,coords (in dstpath will product batch_index data and label nii.gz file)
"""
def read_a_batchData_pickle(path,dstpath):
    df = pd.read_pickle(path)
    print(df.keys())
    print("original_img_shape={}".format(df['original_img_shape']))
    number=len(df['data'])
    for i in range(number):
        seg=np.array(df['seg'][i][0],)
        if(np.sum(seg)>8000):
            data=np.array(df['data'][i][0])
            break
    # produce the patch data to nii.gz
    basename=path.split('/')[-1].split('.')[0]+'_'+str(i)
    # must be (z,x,y)
    data=np.transpose(data,axes=(2,1,0))
    seg=np.transpose(seg,axes=(2,1,0))
    print(data.shape,seg.shape)
    image_nii = sitk.GetImageFromArray(data)
    label_nii = sitk.GetImageFromArray(seg)
    sitk.WriteImage(image_nii, os.path.join(dstpath,basename+'_image.nii.gz'))
    sitk.WriteImage(label_nii, os.path.join(dstpath,basename+'_label.nii.gz'))
    print("nii.gz has saved in the {}".format(dstpath))
    # index,coord
    return i,df['patch_crop_coords'][i]


'''
    function: get nii.gz file from test result named raw_pred_boxes_hold_out_list.pickle
    args: 
        orgindata: image.nii.gz file path
        final_result: path of final result (final_pred_boxes_hold_out_list.pickle)
        raw_result: path of .pickle, index: coord of boxes from origin image, dstpath: where to save results
    return: Rifrac*_index_test_image.nii.gz , Rifrac*_index_test_label.nii.gz
'''
def read_batchData_pickle(orgindata,raw_result,final_result,dstpath,scores=0.1,index=0):
    df=pd.read_pickle(raw_result)
    pid=df[0][1]
    # (z,y,x)
    data=np.load(os.path.join(orgindata,'{}_img.npy'.format(pid)))
    data_nii=sitk.GetImageFromArray(data)
    sitk.WriteImage(data_nii,os.path.join(dstpath,"{}_image.nii.gz".format(pid)))
    #deal with raw_result
    print("deal raw_result:")
    deal_result(df[0], data.shape,scores, gd_box=True)
    # deal with final_result
    print("deal final_result:")
    df = pd.read_pickle(final_result)
    deal_result(df[0],data.shape,scores,gd_box=False)
    print("deal end!")
'''
    function:
        deal with the result of metric
    args: fold_0_test_df.pickle file
'''
def deal_metrics(file):
    df = pd.read_pickle(file)
    print(df.head(1))
    print("columns={},rows={}".format(df.shape[0],df.shape[1]))
    det=df.det_type
    print("det={}".format(det.unique()))
    pred_class=df.pred_class
    print("pred_class={}".format(pred_class.unique()))
    all_p=df[df.class_label==1].shape[0]
    all_n=df[df.class_label==0].shape[0]
    det_tp = df[df.det_type == 'det_tp'].shape[0]
    det_fp = df[df.det_type == 'det_fp'].shape[0]
    print("all_p={},all_n={}".format(all_p,all_n))
    print("TP={},FP={}".format(det_tp,det_fp))
    recall = det_tp / float(all_p)
    precision = det_tp / float(det_tp +det_fp)
    print("Recall={},Precision={}".format(recall,precision))

"""
    class:
        1.product the predict result(.nii.gz) and the ground truth(.nii.gz) [add the boxes to nii.gz file]
        2.caulcate the recall and precision
    args:
        pass
"""
class test_result:
    def __init__(self,path):
        self.path=path
    """
        fucntion: deal the result
        args:
            result: final_result,raw_result []
            origin_shape: shape of origin nii.gz (z,y,x)
    """
    def deal_result(self,dstpath,result, origin_shape, scores, gd_box=False):
        boxes = result[0]['boxes'][0]
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
        det_box_name = "bboxDet_raw.nii.gz" if gd_box else "bboxDet_final.nii.gz"
        if gd_box:
            bbox_gd = np.zeros(origin_shape)
            # print("gt bbox:\n{}".format(gt_bbox_list))
            for value in gt_bbox_list:
                item = value["box_coords"].tolist()
                bbox_gd[item[-2]:item[-1], item[0]:item[2], item[1]:item[3]] = 1
            bbox_gd_nii = sitk.GetImageFromArray(bbox_gd)
            sitk.WriteImage(bbox_gd_nii, os.path.join(dstpath, "{}_bboxGt.nii.gz".format(pid)))
            print("bbox_gd shape is {}".format(bbox_gd.shape))
        #  boxes of detect result
        # print("det_bbox example:\n{}".format(det_bbox_list[-1]))
        bbox_det = np.zeros(origin_shape)
        for value in det_bbox_list:
            item = np.array(value["box_coords"], dtype=int).tolist()
            bbox_det[item[-2]:item[-1], item[0]:item[2], item[1]:item[3]] = 1
        bbox_det_nii = sitk.GetImageFromArray(bbox_det)
        sitk.WriteImage(bbox_det_nii, os.path.join(dstpath, "{}_{}".format(pid, det_box_name)))

    '''
        function: get nii.gz file from test result named raw_pred_boxes_hold_out_list.pickle
        args: 
            orgindata: image.nii.gz file path
            final_result: path of final result (final_pred_boxes_hold_out_list.pickle)
            raw_result: path of .pickle, index: coord of boxes from origin image, dstpath: where to save results
        return: Rifrac*_index_test_image.nii.gz , Rifrac*_index_test_label.nii.gz
    '''
    def read_batchData_pickle(self,scores=0.1):
        raw_result=join(self.path,"raw_pred_boxes_hold_out_list.pickle")
        final_result=join(self.path,"final_pred_boxes_hold_out_list.pickle")
        dstpath=self.path
        df = pd.read_pickle(raw_result)
        pid = df[0][1]
        # (z,y,x)
        data = np.load(os.path.join(self.path, '{}_img.npy'.format(pid)))
        data_nii = sitk.GetImageFromArray(data)
        sitk.WriteImage(data_nii, os.path.join(dstpath, "{}_image.nii.gz".format(pid)))
        # deal with raw_result
        print("deal raw_result:")
        self.deal_result(dstpath,df[0], data.shape, scores, gd_box=True)
        # deal with final_result which is after wbs
        print("deal final_result:")
        df = pd.read_pickle(final_result)
        self.deal_result(dstpath,df[0], data.shape, scores, gd_box=False)
        print("deal end!")

    # cal the recall and precision
    def deal_metrics(self):
        fold_0_test_df=join(self.path,"fold_0_test_df.pickle")
        df = pd.read_pickle(fold_0_test_df)
        all_p = df[df.class_label == 1].shape[0]
        all_n = df[df.class_label == 0].shape[0]
        det_tp = df[df.det_type == 'det_tp'].shape[0]
        det_fp = df[df.det_type == 'det_fp'].shape[0]
        recall = det_tp / float(all_p)
        precision = det_tp / float(det_tp + det_fp)
        print("all_p={},det_tp={}".format(all_p,det_tp))
        print("Recall={},Precision={}".format(recall, precision))

if __name__=="__main__":
    # # show the whole single npy file
    # pid = "421"
    # root_dir = "/Users/jinxiaoqiang/jinxiaoqiang/数据集/Bone/ribfrac/data_npy/"
    # show_npy(root_dir,pid)

    # # read fold_ids.pickle
    # path=os.path.join(cf.pp_data_path, cf.input_df_name)
    # read_fold_ids(path)

    # # show the bacth_data
    # path="/Users/jinxiaoqiang/jinxiaoqiang/DATA/Bone/ribfrac/dataloader_test"
    # pid=428
    # show_batchdata(path,pid)

    # read pickle file
    path = "./examples/498/raw_pred_boxes_hold_out_list.pickle"
    load_seg_from_result(path)

    # # dict2DF
    # path = "/Users/jinxiaoqiang/jinxiaoqiang/DATA/Bone/ribfrac/data_npy/meta_info_RibFrac421.pickle"
    # Dict2df(path)

    # # npy2niigz
    # dstpath='./examples'
    # path='Rib500.npy'
    # dpy2niigz(path,dstpath)
    # dstpath = './examples'
    # path = 'Rib500_seg.npy'
    # dpy2niigz(path, dstpath)
    # path='/media/victoria/9c3e912e-22e1-476a-ad55-181dbde9d785/jinxiaoqiang/rifrac/data_npy/RibFrac500_img.npy'
    # dpy2niigz(path,dstpath)

    # # read a batch_data pickle
    # path="./examples/498/RibFrac498batch.pickle"
    # dst="./examples/498"
    # index,coords=read_a_batchData_pickle(path,dst)
    # print(index,coords)

    # # get batch result of test
    # originpath="/media/victoria/9c3e912e-22e1-476a-ad55-181dbde9d785/jinxiaoqiang/rifrac/data_npy"
    # raw_result="../rifrac_test/fold_0/raw_pred_boxes_hold_out_list.pickle"
    # final_result="../rifrac_test/fold_0/final_pred_boxes_hold_out_list.pickle"
    # index=0
    # # detele the less than scores
    # scores=0.0
    # dstpath="./demo_result"
    # if not os.path.isdir(dstpath):
    #     os.makedirs(dstpath)
    # read_batchData_pickle(originpath,raw_result,final_result,dstpath,scores,index)

    # # deal with metrics
    # file="../rifrac_test/test/fold_0_test_df.pickle"
    # deal_metrics(file)

    # # product the test result
    # path=os.path.join('./examples','499')
    # result=test_result(path)
    # # product nii.gz
    # result.read_batchData_pickle()
    # # return recall and precision
    # result.deal_metrics()
