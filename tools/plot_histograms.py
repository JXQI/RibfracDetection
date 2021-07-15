from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from os.path import join

"""
function: 
    绘制FP,TP分布直方图
args: 
    label,pred,type_list
"""
def plot_prediction_hist(label_list: list, pred_list: list, type_list: list, outfile: str):
    """
    plot histogram of predictions for a specific class.
    :param label_list: list of 1s and 0s specifying whether prediction is a true positive match (1) or a false positive (0).
    False negatives (missed ground truth objects) are artificially added predictions with score 0 and label 1.
    :param pred_list: list of prediction-scores.
    :param type_list: list of prediction-types for stastic-info in title.
    """
    preds = np.array(pred_list)
    labels = np.array(label_list)
    title = outfile.split('/')[-1] + ' count:{}'.format(len(label_list))
    plt.figure()
    plt.yscale('log')
    if 0 in labels:
        plt.hist(preds[labels == 0], alpha=0.3, color='g', range=(0, 1), bins=50, label='false pos.')
    if 1 in labels:
        plt.hist(preds[labels == 1], alpha=0.3, color='b', range=(0, 1), bins=50, label='true pos. (false neg. @ score=0)')

    if type_list is not None:
        fp_count = type_list.count('det_fp')
        fn_count = type_list.count('det_fn')
        tp_count = type_list.count('det_tp')
        pos_count = fn_count + tp_count
        title += ' tp:{} fp:{} fn:{} pos:{}'. format(tp_count, fp_count, fn_count, pos_count)

    plt.legend()
    plt.title(title)
    plt.xlabel('confidence score')
    plt.ylabel('log n')
    # plt.show()
    plt.savefig(outfile)
    # plt.close()

if __name__=='__main__':
    test_df_path="/media/victoria/9c3e912e-22e1-476a-ad55-181dbde9d785/jinxiaoqiang/rifrac/experiment"

    experiments=['RetinaNet_3D_segment_FPD_decoup_pretrain','RetinaNet_3D_segment_FPD_decoup_RPN_pretrain']

    for exper in experiments:
        for mode in ['wbc','raw']:
            if mode=='raw':
                csv_file='all_raw_test_df.csv'
            else:
                csv_file='all_final_test_df.csv'
            raw_test_df=pd.read_csv(join(test_df_path,exper,'val',mode,csv_file))
            # print(raw_test_df.columns)
            label_list=raw_test_df['class_label'].to_list()
            pred_list=raw_test_df['pred_score'].to_list()
            type_list=raw_test_df['det_type'].to_list()
            # print(type(type_list))
            plot_prediction_hist(label_list,pred_list,type_list,exper[22:]+'_'+mode)
        plt.show()