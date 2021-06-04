from torch.utils.data import DataLoader,Dataset
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
from os.path import join
import os
import pandas as pd
from skimage.measure import regionprops
import SimpleITK as sitk
import utils.dataloader_utils as dutils
import utils.exp_utils as utils
# batch generator tools from https://github.com/MIC-DKFZ/batchgenerators
from batchgenerators.dataloading.data_loader import SlimDataLoaderBase
from batchgenerators.transforms.spatial_transforms import MirrorTransform as Mirror
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.dataloading import SingleThreadedAugmenter
from batchgenerators.transforms.spatial_transforms import SpatialTransform
from batchgenerators.transforms.crop_and_pad_transforms import CenterCropTransform
from batchgenerators.transforms.utility_transforms import ConvertSegToBoundingBoxCoordinates

class load_data(Dataset):
    def __init__(self,path,cf,is_training=True):
        self.path=path
        self.image=[join(self.path,i) for i in os.listdir(self.path) if i.endswith(".npz")]
        self.image=self.image[:100]
        self.pid=[i.split('_')[0] for i in os.listdir(self.path) if i.endswith(".npz")]

        p_df = pd.read_pickle(os.path.join(cf.pp_data_path, cf.input_df_name))
        p_df_target=p_df['class_target'].tolist()
        self.class_targets=[]
        p_pid=p_df.pid.tolist()
        for pid in self.pid:
            target_index=p_pid.index(pid)
            target=p_df_target[target_index]
            self.class_targets.append([0 for ii in target if ii>=0])

        my_transforms = []
        if is_training:
            mirror_transform = Mirror(axes=np.arange(cf.dim))
            my_transforms.append(mirror_transform)
            spatial_transform = SpatialTransform(patch_size=cf.patch_size[:cf.dim],
                                                 patch_center_dist_from_border=cf.da_kwargs['rand_crop_dist'],
                                                 do_elastic_deform=cf.da_kwargs['do_elastic_deform'],
                                                 alpha=cf.da_kwargs['alpha'], sigma=cf.da_kwargs['sigma'],
                                                 do_rotation=cf.da_kwargs['do_rotation'],
                                                 angle_x=cf.da_kwargs['angle_x'],
                                                 angle_y=cf.da_kwargs['angle_y'], angle_z=cf.da_kwargs['angle_z'],
                                                 do_scale=cf.da_kwargs['do_scale'], scale=cf.da_kwargs['scale'],
                                                 random_crop=cf.da_kwargs['random_crop'])

            my_transforms.append(spatial_transform)
        else:
            my_transforms.append(CenterCropTransform(crop_size=cf.patch_size[:cf.dim]))
        my_transforms.append(ConvertSegToBoundingBoxCoordinates(cf.dim, get_rois_from_seg_flag=False,
                                                                class_specific_seg_flag=cf.class_specific_seg_flag))
        self.all_transforms = Compose(my_transforms)
    def __len__(self):
        return len(self.image)
    def __getitem__(self, item):
        data_npz = np.load(self.image[item], mmap_mode='r')
        data = data_npz["img"].astype(np.float32)
        seg = data_npz["rois"]
        data = (data - np.mean(data)) / np.std(data).astype(np.float16)
        # data = np.transpose(data, axes=(1, 2, 0))[np.newaxis]
        # seg = np.transpose(seg, axes=(1, 2, 0))

        data=np.array([data])
        seg=np.array([seg.astype(np.uint8)])
        pid=[self.pid[item]]
        target=[self.class_targets[item]]
        # {'data': data, 'seg': seg, 'pid': batch_pids, 'class_target': class_target}
        batch={'data': data, 'seg': seg, 'pid': pid, 'class_target': target}
        # print(data.shape,seg.shape,pid)
        batch=self.all_transforms(**batch)
        return batch

def collate_func(batch_result):
    # print('contat')
    # print(batch_result[0]['data'].shape)
    bb_target,class_target,roi_masks=[],[],[]
    pid,data,seg=[],[],[]
    for item in batch_result:
        bb_target.append(item['bb_target'][0])
        class_target.append(item['class_target'][0])
        roi_masks.append(item['roi_masks'][0])
        pid.append(item['pid'][0])
        data.append(item['data'][0])
        seg.append(item['seg'][0])

    bb_target=np.array(bb_target)
    class_target=np.array(class_target)
    roi_masks=np.array(roi_masks)
    data=np.array(data)
    seg=np.array(seg)
    # print(data.shape,seg.shape)
    # print(bb_target.shape)
    # print(class_target.shape)
    # print(roi_masks.shape)
    return {"data":data,'seg':seg,'pid':pid,'class_target':class_target,'bb_target':bb_target,'roi_masks':roi_masks}

def roi_data_loader(cf,num_works=1,batch_size=16):
    path="/home/victoria/train_data_jinxiaoqiang/medicaldetectiontoolkit/experiments/rifrac_RetinaNet_segment_exp/data/patch_image"
    data_set = load_data(path, cf)
    data_loader = DataLoader(dataset=data_set, num_workers=8, batch_size=16, collate_fn=collate_func)
    return data_loader
'''
function:
   get 3Dpatch and label
'''
class CtTrainDataset:
    def __init__(self,path,crop_size=None,save_dir="./data",cf=None):
        self.path=path
        self.crop_size=crop_size
        self.cf=cf
        self.crop_margin=np.array(self.cf.patch_size)/8. #min distance of ROI center to edge of cropped_patch.
        self.public_id=sorted([x.split("_")[0]
                               for x in os.listdir(path) if x.endswith(".npz")])
        # self.public_id=self.public_id[:11]
        print(self.public_id)
        self.save_dir=save_dir
        if not os.path.isdir(save_dir):
            os.makedirs(self.save_dir)
    # get the centroids of lesion
    @staticmethod
    def _get_pos_centroids(label_arr):
        centroids=[tuple([round(x) for x in prop.centroid])
                         for prop in regionprops(label_arr)]
        return centroids
    def _crop_roi_margin(self,data,seg,centroid):
        roi_anchor_pixel=centroid
        # crop patches of size pre_crop_size, while sampling patches containing foreground with p_fg.
        crop_dims = [dim for dim, ps in enumerate(self.cf.pre_crop_size) if data.shape[dim + 1] > ps]
        sample_seg_center = {}
        for ii in crop_dims:
            low = np.max((self.cf.pre_crop_size[ii] // 2,
                          roi_anchor_pixel[ii] - (self.cf.patch_size[ii] // 2 - self.crop_margin[ii])))
            high = np.min((data.shape[ii + 1] - self.cf.pre_crop_size[ii] // 2,
                           roi_anchor_pixel[ii] + (self.cf.patch_size[ii] // 2 - self.crop_margin[ii])))
            # happens if lesion on the edge of the image. dont care about roi anymore,
            # just make sure pre-crop is inside image.
            if low >= high:
                low = data.shape[ii + 1] // 2 - (data.shape[ii + 1] // 2 - self.cf.pre_crop_size[ii] // 2)
                high = data.shape[ii + 1] // 2 + (data.shape[ii + 1] // 2 - self.cf.pre_crop_size[ii] // 2)
            sample_seg_center[ii] = np.random.randint(low=low, high=high)
        for ii in crop_dims:
            min_crop = int(sample_seg_center[ii] - self.cf.pre_crop_size[ii] // 2)
            max_crop = int(sample_seg_center[ii] + self.cf.pre_crop_size[ii] // 2)
            data = np.take(data, indices=range(min_crop, max_crop), axis=ii + 1)
            seg = np.take(seg, indices=range(min_crop, max_crop), axis=ii)
        return data,seg[np.newaxis]
    # roi
    def _crop_roi(self,arr,centroid):
        roi=np.ones(self.crop_size)*(-1024)
        # src image
        src_beg=[max(0,int(centroid[i]-self.crop_size[i]//2))
                 for i in range(len(centroid))]
        src_end=[min(arr.shape[i],int(centroid[i]+self.crop_size[i]//2))
                 for i in range(len(centroid))]
        dst_beg=[max(0,self.crop_size[i]//2-centroid[i])
                 for i in range(len(centroid))] # max
        dst_end=[min(self.crop_size[i],arr.shape[i]-(centroid[i]-self.crop_size[i]//2))  #边界情况
                 for i in range(len(centroid))] # min
        roi[
            dst_beg[0]:dst_end[0],
            dst_beg[1]:dst_end[1],
            dst_beg[2]:dst_end[2],
        ] = arr[
            src_beg[0]:src_end[0],
            src_beg[1]:src_end[1],
            src_beg[2]:src_end[2],
        ]
        return roi

    # get image patch and label patch
    def _get_patch(self,idx):
        public_id=self.public_id[idx]
        image_path=os.path.join(self.path,f"{public_id}_img.npz")
        data_npz = np.load(image_path, mmap_mode='r')
        image_arr=data_npz["img"]
        label_arr=data_npz["rois"]
        image_arr = np.transpose(image_arr, axes=(1, 2, 0))[np.newaxis]
        label_arr = np.transpose(label_arr, axes=(1, 2, 0))
        # pad data if smaller than pre_crop_size.
        if np.any([image_arr.shape[dim + 1] < ps for dim, ps in enumerate(self.cf.pre_crop_size)]):
            new_shape = [np.max([image_arr.shape[dim + 1], ps]) for dim, ps in enumerate(self.cf.pre_crop_size)]
            image_arr = dutils.pad_nd_image(image_arr, new_shape, mode='constant')
            label_arr = dutils.pad_nd_image(label_arr, new_shape, mode='constant')

        # print(image_arr.shape,label_arr.shape)
        # calcute the rois's centroids
        roi_centroids=self._get_pos_centroids(label_arr)
        # # crop rois
        # image_rois=[self._crop_roi(image_arr,centroid)
        #             for centroid in roi_centroids]
        # label_rois=[self._crop_roi(label_arr,centroid)
        #             for centroid in roi_centroids]
        result=[self._crop_roi_margin(image_arr,label_arr,centroid)
                                for centroid in roi_centroids]
        image_rois,label_rois=[],[]
        for item in result:
            image_rois.append(item[0])
            label_rois.append(item[1])
        return image_rois,label_rois

    # 保存成npy数据，方便读取
    def _save_patch(self):
        for idx in range(len(self.public_id)):
            print("idx={}/{}".format(idx,len(self.public_id)),flush=True)
            image,label=self._get_patch(idx)
            basename=self.public_id[idx]
            for i,data in enumerate(image):
                imgArr=image[i]
                labelArr=label[i]
                # print(i,np.unique(labelArr))
                np.savez_compressed(os.path.join(self.save_dir, basename+"_"+str(i)+".npz"), img=imgArr, rois=labelArr)

def npz2nii(data):
    data=np.load(data)
    image=np.transpose(data['img'][0],axes=(2,0,1))
    label=np.transpose(data['rois'][0],axes=(2,0,1))
    image_nii=sitk.GetImageFromArray(image)
    label_nii=sitk.GetImageFromArray(label)
    sitk.WriteImage(image_nii,"602_image.nii.gz")
    sitk.WriteImage(label_nii,"602_label.nii.gz")

if __name__=='__main__':
    # # group=[{"img":np.array([1]),"label":"ribfrac1","target":[1,2,3]},
    # #        {"img":np.array([2]),"label":"ribfrac2","target":[1]}]
    # # # group=[{"img":1,"label":[1]}]
    # # path = "/Users/jinxiaoqiang/jinxiaoqiang/RifracDetection/deal_data/data"
    # #

    # # 加载数据
    # path='./patch_image'
    # cf_file = utils.import_module("cf", "configs.py")
    # cf = cf_file.configs()
    # cf.pp_data_path = "/Users/jinxiaoqiang/jinxiaoqiang/DATA/Bone/ribfrac/train_image/data_npz"
    # p_df = pd.read_pickle(os.path.join(cf.pp_data_path, cf.input_df_name))
    # data_set=load_data(path,cf)
    # data_loader=DataLoader(dataset=data_set,num_workers=2,batch_size=16,collate_fn=collate_func)
    # for i,item in enumerate(data_loader):
    #     print(item['bb_target'].shape)

    path = "./data/data_segment_npz"
    save_dir="./data/patch_image"
    cf_file = utils.import_module("cf", "configs.py")
    cf = cf_file.configs()
    CTData = CtTrainDataset(path,save_dir=save_dir,cf=cf)
    CTData._save_patch()
    # data="./patch_image/RibFrac96_2.npz"
    # npz2nii(data)