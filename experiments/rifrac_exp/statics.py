from skimage.measure import label,regionprops
import numpy as np
import pandas as pd
import os
import numpy as np
import SimpleITK as sitk
import pickle
from os.path import join
import configs
cf = configs.configs()
from collections import defaultdict
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib
from multiprocessing import Pool
import shutil

# path="/Users/jinxiaoqiang/jinxiaoqiang/DATA/Bone/ribfrac/data_npy"
# rois=[join(path,i) for i in os.listdir(path) if "rois.npy" in i]
#
# feature=[]
# for i in rois:
#     print("deal {}".format(i))
#     item=np.load(i)
#
#     # # test right
#     # item_nii=sitk.GetImageFromArray(item)
#     # sitk.WriteImage(item_nii,"test.nii.gz")
#
#     pre_label=label(item)
#     pred_regions=regionprops(pre_label,item)
#     values=defaultdict(list)
#     for regin in pred_regions:
#         values["bbox"].append(regin.bbox)
#         values["area"].append(regin.area)
#         values["centorid"].append(regin.centroid)
#     values["pid"]=os.path.basename(i).split('_')[0]
#     feature.append(values)
# print(feature)

"""
    function:
        compute the shape and area of rois
"""
class compute_rois:
    def __init__(self,path):
        self.path=path
        self.feature=[]
        self.rois_features_pickle="rois_feature.pickle"
        self.dstpath="./rois_features"
        if not os.path.isdir(self.dstpath):
            os.makedirs(self.dstpath)
    """
        function:
            mul pool to get
        return:
            save result to rois_feature.pickle
    """
    def get_rois_features(self):
        pid_list=[join(self.path, i) for i in os.listdir(path) if "rois.npy" in i]
        pool = Pool(processes=8)
        p1 = pool.map(self.rois_features, enumerate(pid_list), chunksize=1)
        pool.close()
        pool.join()

        # aggrate
        self.aggregate_meto_info()
    """
        function:
            aggrate pid.pickle to roi_feature.pickle
    """
    def aggregate_meto_info(self):
        file=join(self.dstpath,self.rois_features_pickle)
        if os.path.isfile(file):
           os.remove(file)
        for item in os.listdir(self.dstpath):
            item=join(self.dstpath,item)
            self.feature.append(pd.read_pickle(item))
        with open(file,"wb") as handle:
            pickle.dump(self.feature,handle)
    """
        function: 
            compute the center 、bbox、centorid
        return:
            save result to rois_feature.pickle
    """
    def rois_features(self,roi):
        print("deal {}".format(roi))
        item = np.load(roi[1])
        pre_label = label(item)
        pred_regions = regionprops(pre_label, item)
        values = defaultdict(list)
        for regin in pred_regions:
            values["bbox"].append(regin.bbox)
            values["area"].append(regin.area)
            values["centorid"].append(regin.centroid)
        values["pid"] = os.path.basename(roi[1]).split('_')[0]
        file_name=os.path.join(self.dstpath,values["pid"]+".pickle")
        with open(file_name,"wb") as handle:
            pickle.dump(values,handle)
        # print(values)
    """
        function: 
            read the rois_feature.pickle
    """
    def read_rois_pickle(self):
        self.rois_feature=pd.read_pickle(self.rois_features_pickle)
        return self.rois_feature

    """
        function:
            compute the distributed of bbox
        return:
    """
    def distributed_roi(self):
        self.read_rois_pickle()
        self.distributed=defaultdict(list)
        for item in self.rois_feature:
            for index,box in enumerate(item['bbox']):
                zmin,xmin,ymin,zmax,xmax,ymax=box[0],box[1],box[2],box[3],box[4],box[5]
                length,width,high=xmax-xmin,ymax-ymin,zmax-zmin
                self.distributed["shape"].append((length,width,high))
                self.distributed["x_y_ratio"].append(round(length/float(width),1))
            self.distributed["area"].extend((item['area']))
            self.distributed["centriod"].extend(item["centorid"])
        #plot
        self.plot_distributed()
    """
        function: 
            plot distributed_rois
    """
    def plot_distributed(self):
        #x_y ration
        data=self.distributed["x_y_ratio"]
        self.plot_Histogram(data,"x/y ratio")

        # x and y
        shape = self.distributed["shape"]
        x = [item[0] for item in shape]
        y = [item[1] for item in shape]
        z = [item[2] for item in shape]
        self.plot_Histogram(x, "x size")
        self.plot_Histogram(y, "y size")
        self.plot_Histogram(z, "z size")
        self.plot_scatter(x,y,z)
        plt.show()
    """
        function:
            plot Histogram
    """
    def plot_Histogram(self,data,title):
        plt.figure()
        box_wh_unique = list(set(data))
        box_wh_unique.sort()
        box_wh_count = [data.count(i) for i in box_wh_unique]
        # 绘图
        wh_df = pd.DataFrame(box_wh_count, index=box_wh_unique, columns=[title])
        wh_df.plot(kind='bar', color="#55aacc")
    """
        function:
            plot scatter
    """
    def plot_scatter(self,x,y,z):
        plt.figure()
        plt.title("distributed of rois")
        plt.xlabel("x-y:x")
        plt.ylabel("x-y:y")
        plt.scatter(x, y, s=50, c="red", label="x-y size distributed")
        plt.scatter(x, z, s=50, c="green", label="x-z size distributed")
        plt.scatter(y, z, s=50, c="yellow", label="y-z size distributed")
        plt.legend()


if __name__=='__main__':
    path = "/Users/jinxiaoqiang/jinxiaoqiang/DATA/Bone/ribfrac/data_npy"
    handle=compute_rois(path)
    handle.get_rois_features()
    # handle.distributed_roi()