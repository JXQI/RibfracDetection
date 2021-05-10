#!/usr/bin/env python
# Copyright 2018 Division of Medical Image Computing, German Cancer Research Center (DKFZ).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

'''
This preprocessing script loads nrrd files obtained by the data conversion tool: https://github.com/MIC-DKFZ/LIDC-IDRI-processing/tree/v1.0.1
After applying preprocessing, images are saved as numpy arrays and the meta information for the corresponding patient is stored
as a line in the dataframe saved as info_df.pickle.
'''

import os
import SimpleITK as sitk
import numpy as np
from multiprocessing import Pool
import pandas as pd
import numpy.testing as npt
from skimage.transform import resize
import subprocess
import pickle
import copy

import configs
cf = configs.configs()



def resample_array(src_imgs, src_spacing, target_spacing):

    src_spacing = np.round(src_spacing, 3)
    target_shape = [int(src_imgs.shape[ix] * src_spacing[::-1][ix] / target_spacing[::-1][ix]) for ix in range(len(src_imgs.shape))]
    for i in range(len(target_shape)):
        try:
            assert target_shape[i] > 0
        except:
            raise AssertionError("AssertionError:", src_imgs.shape, src_spacing, target_spacing)

    img = src_imgs.astype(float)
    resampled_img = resize(img, target_shape, order=1, clip=True, mode='edge').astype('float32')

    return resampled_img


def pp_patient(inputs):

    ix, path = inputs
    pid = path.split('/')[-1].split('-')[0]
    img = sitk.ReadImage(path)
    img_arr = sitk.GetArrayFromImage(img)
    print('processing {}'.format(pid), img.GetSpacing(), img_arr.shape)
    img_arr = resample_array(img_arr, img.GetSpacing(), cf.target_spacing)
    img_arr = np.clip(img_arr, -300, 1700) #L=700,W=2000
    img_arr = img_arr.astype(np.float32)
    img_arr = (img_arr - np.mean(img_arr)) / np.std(img_arr).astype(np.float16) #Normalize

    df = pd.read_csv(os.path.join(cf.root_dir, cf.csv_file), sep=',')
    df = df[df.public_id == pid]
    # read label.nii.gz
    final_rois = np.zeros_like(img_arr, dtype=np.uint8)
    label=[]
    roi = sitk.ReadImage(os.path.join(cf.raw_label_dir, '{}-label.nii.gz'.format(pid)))
    roi_raters = sitk.GetArrayFromImage(roi).astype(np.uint8)
    for i in np.unique(roi_raters):
        if i>0:
            temp=copy.copy(roi_raters) # notes the copy of list
            temp[temp!=i]=0
            roi_arr = resample_array(temp, roi.GetSpacing(), cf.target_spacing)
            final_rois[roi_arr>0]=i
            label_code=df[df.label_id==i].label_code.values[0]
            if label_code==-1:
                label_code=5
            label.append(label_code)
    mal_labels = np.array(label)
    fg_slices = [ii for ii in np.unique(np.argwhere(final_rois != 0)[:, 0])]    # get slice idx
    assert len(mal_labels)+1 == len(np.unique(final_rois)), [len(mal_labels), np.unique(final_rois), pid]
    np.save(os.path.join(cf.pp_dir, '{}_rois.npy'.format(pid)), final_rois)
    np.save(os.path.join(cf.pp_dir, '{}_img.npy'.format(pid)), img_arr)

    with open(os.path.join(cf.pp_dir, 'meta_info_{}.pickle'.format(pid)), 'wb') as handle:
        meta_info_dict = {'pid': pid, 'class_target': mal_labels, 'spacing': img.GetSpacing(), 'fg_slices': fg_slices}
        pickle.dump(meta_info_dict, handle)

def pp_patient_int2npz(inputs):

    ix, path = inputs
    pid = path.split('/')[-1].split('-')[0]
    img = sitk.ReadImage(path)
    img_arr = sitk.GetArrayFromImage(img)
    print('processing {}'.format(pid), img.GetSpacing(), img_arr.shape)
    img_arr = resample_array(img_arr, img.GetSpacing(), cf.target_spacing)
    img_arr = np.clip(img_arr, -300, 1700) #L=700,W=2000
    img_arr = img_arr.astype(np.int16)
    # img_arr = img_arr.astype(np.float32)
    # img_arr = (img_arr - np.mean(img_arr)) / np.std(img_arr).astype(np.float16) #Normalize

    df = pd.read_csv(os.path.join(cf.root_dir, cf.csv_file), sep=',')
    df = df[df.public_id == pid]
    # read label.nii.gz
    final_rois = np.zeros_like(img_arr, dtype=np.uint8)
    label=[]
    roi = sitk.ReadImage(os.path.join(cf.raw_label_dir, '{}-label.nii.gz'.format(pid)))
    roi_raters = sitk.GetArrayFromImage(roi).astype(np.uint8)
    for i in np.unique(roi_raters):
        if i>0:
            temp=copy.copy(roi_raters) # notes the copy of list
            temp[temp!=i]=0
            roi_arr = resample_array(temp, roi.GetSpacing(), cf.target_spacing)
            final_rois[roi_arr>0]=i
            label_code=df[df.label_id==i].label_code.values[0]
            if label_code==-1:
                label_code=5
            label.append(label_code)
    mal_labels = np.array(label)
    fg_slices = [ii for ii in np.unique(np.argwhere(final_rois != 0)[:, 0])]    # get slice idx
    assert len(mal_labels)+1 == len(np.unique(final_rois)), [len(mal_labels), np.unique(final_rois), pid]
    # np.save(os.path.join(cf.pp_dir, '{}_rois.npy'.format(pid)), final_rois)
    np.savez_compressed(os.path.join(cf.pp_dir, '{}_img.npz'.format(pid)), img=img_arr,rois=final_rois)
    print(cf.pp_dir)
    with open(os.path.join(cf.pp_dir, 'meta_info_{}.pickle'.format(pid)), 'wb') as handle:
        meta_info_dict = {'pid': pid, 'class_target': mal_labels, 'spacing': img.GetSpacing(), 'fg_slices': fg_slices}
        pickle.dump(meta_info_dict, handle)

def aggregate_meta_info(exp_dir):

    files = [os.path.join(exp_dir, f) for f in os.listdir(exp_dir) if 'meta_info' in f]
    df = pd.DataFrame(columns=['pid', 'class_target', 'spacing', 'fg_slices'])
    for f in files:
        with open(f, 'rb') as handle:
            df.loc[len(df)] = pickle.load(handle)

    df.to_pickle(os.path.join(exp_dir, 'info_df.pickle'),protocol=4)
    print ("aggregated meta info to df with length", len(df))


if __name__ == "__main__":
    paths = [os.path.join(cf.raw_data_dir, ii) for ii in os.listdir(cf.raw_data_dir) if ii.endswith(".nii.gz")]
    print(cf.pp_dir)
    if not os.path.exists(cf.pp_dir):
        os.mkdir(cf.pp_dir)
    paths=paths[0:2]
    pool = Pool(processes=2)
    p1 = pool.map(pp_patient_int2npz, enumerate(paths), chunksize=1)
    pool.close()
    pool.join()
    # for i in enumerate(paths):
    #     pp_patient(i)

    aggregate_meta_info(cf.pp_dir)
    subprocess.call('cp {} {}'.format(os.path.join(cf.pp_dir, 'info_df.pickle'), os.path.join(cf.pp_dir, 'info_df_bk.pickle')), shell=True)