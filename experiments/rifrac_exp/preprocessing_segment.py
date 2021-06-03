import SimpleITK as sitk
import numpy as np
from skimage.measure import regionprops, label
from os.path import join
import os
from skimage.transform import resize
import pandas as pd
import copy
import pickle
import time
from multiprocessing import Pool
import subprocess


# good,but drop some information
def Region_3D(soucre, label):
    lowerThreshold = 217
    uppertThreshold = 1404
    region_val = 1
    index = (329, 346, 197)  # sig cor axial (x,y,z)
    pix_value = 711

    # CT self
    source = "/Users/jinxiaoqiang/jinxiaoqiang/DATA/Bone/ribfrac/train_image/image/RibFrac3-image.nii.gz"
    source = sitk.ReadImage(source)
    source_arry = sitk.GetArrayFromImage(source)

    label = "/Users/jinxiaoqiang/jinxiaoqiang/DATA/Bone/ribfrac/train_image/label/RibFrac3-label.nii.gz"
    label = sitk.ReadImage(label)
    label_arry = sitk.GetArrayFromImage(label)
    print(np.unique(label_arry))
    label_arry[label_arry > 1] = 1
    print(np.unique(label_arry))
    new_mask = source_arry * label_arry
    pix_value = int(np.max(new_mask))
    print(pix_value)
    conda = np.where(new_mask == pix_value)  # (z,y,x)

    index = (int(conda[2][0]), int(conda[1]), int(conda[0]))
    print(index)
    # get pix value of index
    source.SetPixel(index, pix_value)
    # caster
    caster = sitk.CastImageFilter()
    # output type
    caster.SetOutputPixelType(sitk.sitkInt8)
    smoothing = sitk.CurvatureFlowImageFilter()
    smoothing.SetNumberOfIterations(5)
    smoothing.SetTimeStep(0.125)
    # region growing filter
    connectedThreshold = sitk.ConnectedThresholdImageFilter()
    connectedThreshold.SetLower(lowerThreshold)
    connectedThreshold.SetUpper(uppertThreshold)
    # set region
    connectedThreshold.SetReplaceValue(region_val)
    # a sed point
    connectedThreshold.SetSeed(index)

    # process
    case1 = smoothing.Execute(source)
    case1 = connectedThreshold.Execute(case1)
    case1 = caster.Execute(case1)

    path = "/Users/jinxiaoqiang/Desktop/EXPERIMENT/rifrac3_test_mask.nii.gz"
    sitk.WriteImage(case1, path)

    # morphological operation
    # IteratorType=sitk.BinaryMorphologicalOpeningImageFilter()
    # IteratorType=sitk.BinaryMorphologicalClosingImageFilter()
    IteratorType = sitk.DilateObjectMorphologyImageFilter()
    element_radius = 3
    IteratorType.SetKernelRadius(element_radius)
    case1_open = IteratorType.Execute(case1)
    case1_open = IteratorType.Execute(case1_open)

    path = "/Users/jinxiaoqiang/Desktop/EXPERIMENT/rifrac3_test_mask_open.nii.gz"
    sitk.WriteImage(case1_open, path)


# 高斯滤波，对图像进行滤波
def Gaussian_Filter(input, Sigma=1):
    Gaussian_filter = sitk.RecursiveGaussianImageFilter()
    Gaussian_filter.SetSigma(Sigma)
    Gaussian_filter.SetDirection(0)
    Gaussian_result = Gaussian_filter.Execute(input)
    Gaussian_filter.SetDirection(1)
    Gaussian_result = Gaussian_filter.Execute(Gaussian_result)
    Gaussian_filter.SetDirection(2)
    # get information
    # print(Gaussian_filter.GetDirection(), Gaussian_filter.GetSigma(), Gaussian_filter.GetOrder())
    Gaussian_result = Gaussian_filter.Execute(Gaussian_result)

    return Gaussian_result


# 二值化图像，利用最简单的阈值方法
def Binary_Image(input, thread=200):
    lowerThreshold = thread
    thread_filter = sitk.BinaryThresholdImageFilter()
    thread_filter.SetOutsideValue(0)
    thread_filter.SetLowerThreshold(lowerThreshold)
    # print(thread_filter.GetLowerThreshold())
    thread_result = thread_filter.Execute(input)

    return thread_result


# 和骨折的标签做对比，计算出丢失的骨折部分，骨头部分无法计算，暂时忽略
# 输入骨折的标签和肋骨部分的mask,输出丢失的数目和所有骨折数目
def Differ_Fraction_between_Ribfrac(fraction, ribfrac):
    fraction_arry = sitk.GetArrayFromImage(fraction)
    fraction_arry[fraction_arry >= 1] = 1
    ribfrac_arry = sitk.GetArrayFromImage(ribfrac)

    common_area = fraction_arry * ribfrac_arry
    # 计算label连通域的个数
    _, fraction_arry_num = label(fraction_arry, return_num=True)
    _, common_area_num = label(common_area, return_num=True)

    return (fraction_arry_num - common_area_num, fraction_arry_num)


# 得到连通域中目标最大的部分（脊柱）,input的是二值化图像
def Spine_Commpent(input, Number=4):
    # connect
    cc = sitk.ConnectedComponent(input)
    stats = sitk.LabelIntensityStatisticsImageFilter()
    stats.SetGlobalDefaultNumberOfThreads(Number)
    stats.Execute(cc, input)
    maxlabel = 0
    maxsize = 0
    for l in stats.GetLabels():
        size = stats.GetPhysicalSize(l)
        if maxsize < size:
            maxlabel = l
            maxsize = size
    labelmaskimage = sitk.GetArrayFromImage(cc)
    outmask = labelmaskimage.copy()
    outmask[labelmaskimage == maxlabel] = 1
    outmask[labelmaskimage != maxlabel] = 0
    # convert to itk
    outmasksitk = sitk.GetImageFromArray(outmask)
    outmasksitk.SetSpacing(input.GetSpacing())
    outmasksitk.SetOrigin(input.GetOrigin())
    outmasksitk.SetDirection(input.GetDirection())

    return outmasksitk


# 得到肋骨部分，二值化图像减去脊柱部分
def RibFrac_Binary(Binary_input, Spine):
    Bone_arry = sitk.GetArrayFromImage(Binary_input)
    Spine_arry = sitk.GetArrayFromImage(Spine)
    rifrac_arry = Bone_arry - Spine_arry
    rifracmasksitk = sitk.GetImageFromArray(rifrac_arry)
    rifracmasksitk.SetSpacing(Spine.GetSpacing())
    rifracmasksitk.SetOrigin(Spine.GetOrigin())
    rifracmasksitk.SetDirection(Spine.GetDirection())

    return rifracmasksitk


# 对肋骨部分进行形态学操作，保留完整的肋骨信息,输入的是仅仅只有肋骨的部分
def Morphology_Ribfrac(input, radius=3):
    IteratorType = sitk.DilateObjectMorphologyImageFilter()
    element_radius = radius
    IteratorType.SetKernelRadius(element_radius)
    Morphology_result = IteratorType.Execute(input)
    Morphology_result = IteratorType.Execute(Morphology_result)

    return Morphology_result


# 得到最终CT图像中只保留肋骨部分的原图
def Origin_Ribfrac(input, mask):
    input_arry = sitk.GetArrayFromImage(input)
    mask_arry = sitk.GetArrayFromImage(mask)
    origin_Ribfrac_arry = input_arry * mask_arry
    origin_Ribfrac_arry = origin_Ribfrac_arry.astype(np.int32)
    origin_RibfracItk = sitk.GetImageFromArray(origin_Ribfrac_arry)
    origin_RibfracItk.SetDirection(input.GetDirection())
    origin_RibfracItk.SetOrigin(input.GetOrigin())
    origin_RibfracItk.SetSpacing(input.GetSpacing())

    return origin_RibfracItk


def segment_singal_CT(image, label, Is_store=False, path=None):
    # get input
    input = sitk.ReadImage(image)
    label = sitk.ReadImage(label)

    # Blurring by Gaussian
    Gaussian_output = Gaussian_Filter(input)
    # Binary by Thread
    Binary_output = Binary_Image(Gaussian_output)
    # get spine
    Spine_output = Spine_Commpent(Binary_output)
    # get only rifrac part
    Ribfrac_output = RibFrac_Binary(Binary_output, Spine_output)
    # get Morphology Ribfrac part
    Morphology_output = Morphology_Ribfrac(Ribfrac_output)

    # get Ribfrac that load mask on CT
    RibFrac_OriginCt = Origin_Ribfrac(input, Morphology_output)

    # compute number of the drop labels
    drop_labelNum = Differ_Fraction_between_Ribfrac(label, Morphology_output)

    # reslut dict
    output = [Gaussian_output, Binary_output, Spine_output, Ribfrac_output, Morphology_output, RibFrac_OriginCt]
    outname = ["Gaussian_output", "Binary_output", "Spine_output", "Ribfrac_output", "Morphology_output",
               "RibFrac_OriginCt"]
    if Is_store:
        for name, data in zip(outname, output):
            name = name + ".nii.gz"
            name = join(path, name)
            sitk.WriteImage(data, name)

    # 返回 origin,mask,drop_num
    return RibFrac_OriginCt, Morphology_output, drop_labelNum, label


# 查看分割结果，生成中间的过程
def debug_segment():
    image = "/Users/jinxiaoqiang/jinxiaoqiang/DATA/Bone/ribfrac/train_image/image/RibFrac96-image.nii.gz"
    label_mask = "/Users/jinxiaoqiang/jinxiaoqiang/DATA/Bone/ribfrac/train_image/label/RibFrac96-label.nii.gz"
    dir_path = "/Users/jinxiaoqiang/Desktop/EXPERIMENT"
    Ribfrac_CT, Ribfrac_Mask, Drop_num, Label = segment_singal_CT(image, label_mask, True, dir_path)

    Rifrac_CT_name = join(dir_path, "RibFrac_CT.nii.gz")
    Ribfrac_Mask_name = join(dir_path, "RibFrac_Mask.nii.gz")
    sitk.WriteImage(Ribfrac_CT, Rifrac_CT_name)
    sitk.WriteImage(Ribfrac_Mask, Ribfrac_Mask_name)
    print("Drop [{}] labels in all [{}] labels".format(Drop_num[0], Drop_num[1]))


def resample_array(src_imgs, src_spacing, target_spacing,anti_aliasing=True,mode="edge"):
    src_spacing = np.round(src_spacing, 3)
    target_shape = [int(src_imgs.shape[ix] * src_spacing[::-1][ix] / target_spacing[::-1][ix]) for ix in
                    range(len(src_imgs.shape))]
    for i in range(len(target_shape)):
        try:
            assert target_shape[i] > 0
        except:
            raise AssertionError("AssertionError:", src_imgs.shape, src_spacing, target_spacing)

    img = src_imgs.astype(float)
    # resampled_img = resize(img, target_shape, order=1, clip=True, mode='edge').astype('float32')
    resampled_img = resize(img, target_shape, order=1, clip=True, mode=mode,anti_aliasing=anti_aliasing).astype(np.int16)
    return resampled_img


# 生成单个病人的数据
def deal_singal_patient(pid):
    index, pid = pid

    image_path = join(path, "image", pid + "-image.nii.gz")
    label_path = join(path, "label", pid + "-label.nii.gz")
    # print(image_path, label_path)

    segment_begin = time.time()
    # prodcuct ribfrac CT
    Ribfrac, Mask, Drop_labelNum, roi = segment_singal_CT(image_path, label_path)
    segment_end = time.time()
    # print("segmentation done!spend time=[{}]".format(segment_end - segment_begin))

    # new image
    Ribfrac_arry = sitk.GetArrayFromImage(Ribfrac)
    Ribfrac_arry = resample_array(Ribfrac_arry, Ribfrac.GetSpacing(), target_spacing=target_spacing)
    Ribfrac_arry = np.clip(Ribfrac_arry, -300, 1700)  # L=700,W=2000
    # Ribfrac_arry = Ribfrac_arry.astype(np.int16)
    Mask_arry = sitk.GetArrayFromImage(Mask)
    Mask_arry = resample_array(Mask_arry, Ribfrac.GetSpacing(), target_spacing=target_spacing)
    df = pd.read_csv(os.path.join(path, csv_file), sep=',')
    df = df[df.public_id == pid]

    # read label.nii.gz
    begin=time.time()
    roi_raters = sitk.GetArrayFromImage(roi).astype(np.uint8)
    final_rois = np.zeros_like(Ribfrac_arry, dtype=np.uint8)
    label = []
    for i in np.unique(roi_raters):
        if i > 0:
            temp = copy.copy(roi_raters)  # notes the copy of list
            temp[temp != i] = 0
            roi_arr = resample_array(temp, roi.GetSpacing(), target_spacing)
            final_rois[roi_arr > 0] = i
            label_code = df[df.label_id == i].label_code.values[0]
            if label_code == -1:
                label_code = 5
            label.append(label_code)
    # TODO: 上述生成数据的方式太慢，需要改进
    # label_nums=np.unique(roi_raters)
    # final_rois = resample_array(roi_raters, roi.GetSpacing(), target_spacing)
    # final_rois=final_rois.astype(np.uint8)
    # label=df.label_code.values
    # label=np.array(label)
    # label[label==-1]=5
    # label=label[label>0]
    # label_nums=label_nums[label_nums>0].tolist()
    # print(len(label_nums),len(label))
    # print("spent time={}".format(time.time()-begin))

    mal_labels = np.array(label)
    fg_slices = [ii for ii in np.unique(np.argwhere(final_rois != 0)[:, 0])]  # get slice idx
    #
    # assert len(mal_labels) + 1 == len(np.unique(final_rois)), [len(mal_labels), np.unique(final_rois), pid]
    # np.save(os.path.join(cf.pp_dir, '{}_rois.npy'.format(pid)), final_rois)
    np.savez_compressed(os.path.join(pp_dir, '{}_img.npz'.format(pid)), img=Ribfrac_arry, rois=final_rois,mask=Mask_arry)
    # print("convert to npz done ! spent time [{}]".format(time.time() - segment_end))
    with open(os.path.join(pp_dir, 'meta_info_{}.pickle'.format(pid)), 'wb') as handle:
        meta_info_dict = {'pid': pid, 'class_target': mal_labels, 'spacing': Ribfrac.GetSpacing(),
                          'fg_slices': fg_slices}
        pickle.dump(meta_info_dict, handle)

    # save Drop num
    Drop_labelNum=list(Drop_labelNum)
    Drop_label={pid:pd.Series(Drop_labelNum,index=["drop","all"])}
    Drop_label=pd.DataFrame(Drop_label)
    Drop_label=Drop_label.T
    Drop_label.to_csv(join(path,pp_dir,pid+'.csv'))

    print("deal process [{}]".format(index))



def aggregate_meta_info(exp_dir):
    files = [os.path.join(exp_dir, f) for f in os.listdir(exp_dir) if 'meta_info' in f]
    df = pd.DataFrame(columns=['pid', 'class_target', 'spacing', 'fg_slices'])
    for f in files:
        with open(f, 'rb') as handle:
            df.loc[len(df)] = pickle.load(handle)

    df.to_pickle(os.path.join(exp_dir, 'info_df.pickle'), protocol=4)
    print("aggregated meta info to df with length", len(df))

    # csv file
    csvfile=[os.path.join(exp_dir, f) for f in os.listdir(exp_dir) if 'csv' in f]
    print(csvfile)
    df=[pd.read_csv(i) for i in csvfile]
    df=pd.concat(df)
    print(df)
    df.to_csv(join(path,pp_dir,'Drop_label.csv'))




target_spacing = (0.7, 0.7, 1.25)
path = "/media/victoria/9c3e912e-22e1-476a-ad55-181dbde9d785/jinxiaoqiang/rifrac/ribfrac_train/"
csv_file = "ribfrac-train-info.csv"
dir_path="/home/victoria/train_data_jinxiaoqiang/ribfrac_train/"
pp_dir = join(dir_path, "data_segment_npz")
if not os.path.exists(pp_dir):
    os.makedirs(pp_dir)

# pids = [os.path.basename(item).split("-")[0] for item in os.listdir(join(path, "image")) if
#         item.endswith(".nii.gz")]
pids=['RibFrac22', 'RibFrac237', 'RibFrac279', 'RibFrac181', 'RibFrac385', \
 'RibFrac102', 'RibFrac119', 'RibFrac286', 'RibFrac23', 'RibFrac314', \
 'RibFrac96', 'RibFrac186', 'RibFrac353', 'RibFrac164', 'RibFrac359', \
 'RibFrac406', 'RibFrac125', 'RibFrac367', 'RibFrac232', 'RibFrac134', \
 'RibFrac187', 'RibFrac42', 'RibFrac321', 'RibFrac264', 'RibFrac330', \
 'RibFrac41', 'RibFrac80', 'RibFrac35', 'RibFrac230', 'RibFrac221', 'RibFrac84',\
 'RibFrac320', 'RibFrac207', 'RibFrac352', 'RibFrac30', 'RibFrac349', \
 'RibFrac12', 'RibFrac328', 'RibFrac224', 'RibFrac335', 'RibFrac302', \
 'RibFrac270', 'RibFrac5', 'RibFrac408', 'RibFrac25', 'RibFrac197', \
 'RibFrac52', 'RibFrac238', 'RibFrac249', 'RibFrac31', 'RibFrac316', \
 'RibFrac364', 'RibFrac300', 'RibFrac205', 'RibFrac135', 'RibFrac203',\
 'RibFrac123', 'RibFrac145', 'RibFrac258', 'RibFrac304', 'RibFrac294', \
 'RibFrac412', 'RibFrac227', 'RibFrac313', 'RibFrac272', 'RibFrac95', \
 'RibFrac117']
# pids=pids[:1]
print(len(pids))
if __name__ == '__main__':
    pool = Pool(processes=3)
    p1 = pool.map(deal_singal_patient, enumerate(pids), chunksize=1)
    pool.close()
    pool.join()

    aggregate_meta_info(pp_dir)
    subprocess.call(
        'cp {} {}'.format(os.path.join(pp_dir, 'info_df.pickle'), os.path.join(pp_dir, 'info_df_bk.pickle')),
        shell=True)

