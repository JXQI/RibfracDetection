import pandas as pd
import os
import numpy as np
import SimpleITK as sitk
import pickle
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

def read_pick(path):
    df=pd.read_pickle(path)
    print(len(df[0][0]['boxes']))
    print(df[0][0]['boxes'][0][-1])
    seg=np.array(df[0][0]['seg_preds'][0][0],dtype=np.int8)
    print(seg.shape)
    gt_label=np.zeros(seg.shape)
    coordid=df[0][0]['boxes'][0][-1]['box_coords']
    gt_label[coordid[-2]:coordid[-1],coordid[3]:coordid[2],coordid[1]:coordid[0]]=1
    seg[seg>0]=2
    print(np.unique(seg))
    np.save("Rib500_seg.npy",seg)
    np.save("Rib500.npy",gt_label)

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
def read_batchData_pickle(path,dstpath):
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
        path: path of .pickle, index: coord of boxes from origin image, dstpath: where to save results
    return: Rifrac*_index_test_image.nii.gz , Rifrac*_index_test_label.nii.gz
'''
def read_batchData_pickle(orgindata,path,dstpath,index=0):
    df=pd.read_pickle(path)
    result_dict=df[0]
    result=result_dict[0]
    pid=result_dict[1]
    print(pid,result.keys())

    # [box_0, ...]
    boxes=result['boxes'][0]
    num_gt=0
    det_num=0
    gt_bbox_list=[] # gt bbox (y1,y2,x1,x2,z1,z2)
    det_bbox_list=[] # test result bbox
    for i in range(len(boxes)):
        if(boxes[i]["box_type"]=='gt'):
            num_gt+=1
            gt_bbox_list.append(boxes[i])
        elif(boxes[i]["box_type"]=='det'):
            det_num+=1
            det_bbox_list.append(boxes[i])
        else:
            print(boxes[i]["box_type"])
    print("box_num={},gt_num={},det_num={}".format(len(boxes),num_gt,det_num))
    # set gt_bbox to origin shape , and produce new pid_boxGt_nii.gz
    # (z,y,x)
    data=np.load(os.path.join(orgindata,'{}_img.npy'.format(pid)))
    data_nii=sitk.GetImageFromArray(data)
    sitk.WriteImage(data_nii,os.path.join(dstpath,"{}_image.nii.gz".format(pid)))
    bbox_gd=np.zeros(data.shape)

    print("gt bbox:\n{}".format(gt_bbox_list))
    for value in gt_bbox_list:
        item=value["box_coords"].tolist()
        bbox_gd[item[-2]:item[-1],item[0]:item[2],item[1]:item[3]]=1
    bbox_gd_nii=sitk.GetImageFromArray(bbox_gd)
    sitk.WriteImage(bbox_gd_nii,os.path.join(dstpath,"{}_bboxGt.nii.gz".format(pid)))
    print("bbox_gd shape is {}".format(bbox_gd.shape))
    # seg(b, 1, y, x, (z))
    segs=np.array(result['seg_preds'][0][0],dtype=np.int8) #TODO: why?
    # must z,x,y
    segs=np.transpose(segs,axes=(2,1,0))
    print(segs.shape)

    label_nii = sitk.GetImageFromArray(segs)
    sitk.WriteImage(label_nii, os.path.join(dstpath, pid + '_test.nii.gz'))

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

    # # read pickle file
    # path = "../rifrac_test/fold_0/raw_pred_boxes_hold_out_list.pickle"
    # read_pick(path)

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
    # path="../rifrac_test/fold_0/RibFrac500batch.pickle"
    # dst="./examples"
    # index,coords=read_batchData_pickle(path,dst)
    # print(index,coords)

    # get batch result of test
    originpath="/media/victoria/9c3e912e-22e1-476a-ad55-181dbde9d785/jinxiaoqiang/rifrac/data_npy"
    path="../rifrac_test/fold_0/raw_pred_boxes_hold_out_list.pickle"
    index=0
    dstpath="./examples"
    read_batchData_pickle(originpath,path,dstpath,index)