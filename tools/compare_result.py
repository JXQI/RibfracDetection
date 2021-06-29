import matplotlib.pyplot as plt
import numpy as np

from experiments.rifrac_exp.froc import evaluate,DEFAULT_KEY_FP
import pandas as pd
from os.path import join

def plot_froc(items,labels):
    y=[]
    for i in items:
        result=evaluate(pd.read_csv(i))
        y.append(result["detection"]['key_recall'])
    x=range(len(DEFAULT_KEY_FP))
    plt.xlim((x[0],x[-1]))
    plt.ylim((0,1))
    plt.grid()
    plt.xlabel('Average number of false positives pre scan')
    plt.ylabel("Sensitivity")
    for i in range(len(y)):
        plt.plot(x,y[i],label=labels[i])
    # 设置坐标值刻度
    plt.yticks(np.arange(0,1,0.1))
    plt.xticks(x,DEFAULT_KEY_FP)
    plt.legend()
    plt.show()


if __name__=="__main__":
    labels=["ori","segment_bak","segment","segment_tunepara"]
    results=[]
    mode='val'
    postprocess_way='wbc'
    detection_result="all_final_test_df.csv"
    path="/media/victoria/9c3e912e-22e1-476a-ad55-181dbde9d785/jinxiaoqiang/rifrac/experiment"
    for i in labels:
        result_path=join(path,"RetinaNet_3D_"+i,mode,postprocess_way,detection_result)
        results.append(result_path)
    plot_froc(results,labels)
