"""
function:
    in order to save memory,so convert npy to npz
"""

import os
from os.path import join
import numpy as np
from  tqdm import tqdm
import shutil
from multiprocessing import Pool
import time

def npy2npyz(filename):
    filename=filename[1]
    print(os.path.basename(filename).split('.')[0])
    if ".npy" in filename:
        data=np.load(filename)
        filename = os.path.basename(filename).split('.')[0] + '.npz'
        np.savez_compressed(join(dstpath,filename),img=data)
    elif "pickle" in filename:
        shutil.copy(filename,dstpath)

dstpath="/home/victoria/train_data_jinxiaoqiang/all_data/data_npy"
srcpath="/media/victoria/9c3e912e-22e1-476a-ad55-181dbde9d785/jinxiaoqiang/rifrac/rifrac_all_data/data_npy"

def test_readSpeed(path):
    paths=[join(path,item) for item in os.listdir(path) if item.endswith(".npy") or item.endswith(".npz")]
    beign=time.time()
    for i in tqdm(paths):
        a=np.load(i,mmap_mode='r+')
    end=time.time()
    print(end-beign)

if __name__=="__main__":
    # # convert npy to npz
    # if not os.path.isdir(dstpath):
    #     os.makedirs(dstpath)
    # paths=[join(srcpath,item) for item in os.listdir(srcpath)]
    # pool = Pool(processes=20)
    # p1 = pool.map(npy2npyz, enumerate(paths), chunksize=1)
    # pool.close()
    # pool.join()
    # test_readSpeed(srcpath)
    test_readSpeed(dstpath)
