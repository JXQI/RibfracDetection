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
    seg=np.array(df[0][0]['seg_preds'][0][0],dtype=np.int8)
    print(seg.shape)
    seg[seg>=0.5]=1
    seg[seg<0.5]=0
    print(np.unique(seg))
    np.save("Rib500.npy",seg)

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
    path = "../rifrac_test/fold_0/raw_pred_boxes_hold_out_list.pickle"
    read_pick(path)

    # # dict2DF
    # path = "/Users/jinxiaoqiang/jinxiaoqiang/DATA/Bone/ribfrac/data_npy/meta_info_RibFrac421.pickle"
    # Dict2df(path)

    # npy2niigz
    dstpath='./examples'
    path='Rib500.npy'
    dpy2niigz(path,dstpath)
    path='/media/victoria/9c3e912e-22e1-476a-ad55-181dbde9d785/jinxiaoqiang/rifrac/data_npy/RibFrac500_img.npy'
    dpy2niigz(path,dstpath)