import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from skimage.measure import label, regionprops
from tqdm import tqdm


__all__ = ["froc", "plot_froc", "evaluate"]


# detection key FP values
DEFAULT_KEY_FP = (0.5, 1, 2, 4, 8)


pd.set_option("display.precision", 6)


def _froc_single_thresh(df_list, num_gts, p_thresh, iou_thresh):
    """
    Calculate the FROC for a single confidence threshold.

    Parameters
    ----------
    df_list : list of pandas.DataFrame
        List of Pandas DataFrame of prediction metrics.
    num_gts : list of int
        List of number of GT in each volume.
    p_thresh : float
        The probability threshold of positive predictions.
    iou_thresh : float
        The IoU threshold of predictions being considered as "hit".

    Returns
    -------
    fp : float
        False positives per scan for this threshold.
    recall : float
        Recall rate for this threshold.
    """
    EPS = 1e-8

    total_gt = num_gts
    img_nums = len(df_list.pid.unique())
    # collect all predictions above the probability threshold
    df_pos_pred = df_list[df_list.pred_score>p_thresh]

    # calculate total true positives
    # total_tp = sum([len(df.loc[df["max_iou"] > iou_thresh, "hit_label"]\
    #     .unique()) for df in df_pos_pred])
    total_tp=len(df_pos_pred[df_pos_pred["class_label"]==1])
    # calculate total false positives
    # total_fp = sum([len(df) - len(df.loc[df["max_iou"] > iou_thresh])
    #     for df in df_pos_pred])
    total_fp=len(df_pos_pred[df_pos_pred["class_label"]==0])
    # fp = (total_fp + EPS) / (len(df_list) + EPS)
    fp=(total_fp+EPS)/(img_nums+EPS)
    recall = (total_tp + EPS) / (total_gt + EPS)
    print(img_nums,fp,recall)
    return fp, recall


def _interpolate_recall_at_fp(fp_recall, key_fp):
    """
    Calculate recall at key_fp using interpolation.

    Parameters
    ----------
    fp_recall : pandas.DataFrame
        DataFrame of FP and recall.
    key_fp : float
        Key FP threshold at which the recall will be calculated.

    Returns
    -------
    recall_at_fp : float
        Recall at key_fp.
    """
    # get fp/recall interpolation points
    fp_recall_less_fp = fp_recall.loc[fp_recall.fp <= key_fp]
    fp_recall_more_fp = fp_recall.loc[fp_recall.fp >= key_fp]

    # if key_fp < min_fp, recall = 0
    if len(fp_recall_less_fp) == 0:
        return 0

    # if key_fp > max_fp, recall = max_recall
    if len(fp_recall_more_fp) == 0:
        return fp_recall.recall.max()

    fp_0 = fp_recall_less_fp["fp"].values[-1]
    fp_1 = fp_recall_more_fp["fp"].values[0]
    recall_0 = fp_recall_less_fp["recall"].values[-1]
    recall_1 = fp_recall_more_fp["recall"].values[0]
    recall_at_fp = recall_0 + (recall_1 - recall_0)\
        * ((key_fp - fp_0) / (fp_1 - fp_0 + 1e-8))

    return recall_at_fp


def _get_key_recall(fp, recall, key_fp_list):
    """
    Calculate recall at a series of FP threshold.

    Parameters
    ----------
    fp : list of float
        List of FP at different probability thresholds.
    recall : list of float
        List of recall at different probability thresholds.
    key_fp_list : list of float
        List of key FP values.

    Returns
    -------
    key_recall : list of float
        List of key recall at each key FP.
    """
    fp_recall = pd.DataFrame({"fp": fp, "recall": recall}).sort_values("fp")
    key_recall = [_interpolate_recall_at_fp(fp_recall, key_fp)
        for key_fp in key_fp_list]

    return key_recall


def froc(df_list, num_gts, iou_thresh=0.1, key_fp=DEFAULT_KEY_FP):
    """
    Calculate the FROC curve.

    Parameters
    df_list : list of pandas.DataFrame
        List of prediction metrics.
    num_gts : list of int
        List of number of GT in each volume.
    iou_thresh : float
        The IoU threshold of predictions being considered as "hit".
    key_fp : tuple of float
        The key false positive per scan used in evaluating the sensitivity
        of the model.

    Returns
    -------
    fp : list of float
        List of false positives per scan at different probability thresholds.
    recall : list of float
        List of recall at different probability thresholds.
    key_recall : list of float
        List of key recall corresponding to key FPs.
    avg_recall : float
        Average recall at key FPs. This is the evaluation metric we use
        in the detection track.
    """
    fp_recall = [_froc_single_thresh(df_list, num_gts, p_thresh, iou_thresh)
        for p_thresh in np.arange(0.1, 1, 0.01)]
    fp = [x[0] for x in fp_recall]
    recall = [x[1] for x in fp_recall]
    key_recall = _get_key_recall(fp, recall, key_fp)
    avg_recall = np.mean(key_recall)

    return fp, recall, key_recall, avg_recall


def plot_froc(fp, recall):
    """
    Plot the FROC curve.

    Parameters
    ----------
    fp : list of float
        List of false positive per scans at different confidence thresholds.
    recall : list of float
        List of recall at different confidence thresholds.
    """
    _, ax = plt.subplots()
    ax.plot(fp, recall)
    ax.set_title("FROC")
    plt.savefig("./examples/froc.jpg")


def evaluate(det_results):
    """
    Evaluate predictions against the ground-truth.

    Parameters
    ----------
    gt_dir : str
        The ground-truth nii directory.
    pred_dir : str
        The prediction nii directory.

    Returns
    -------
    eval_results : dict
        Dictionary containing detection and classification results.
    """
    num_gts=len(det_results[det_results.class_label==1])
    det_results=det_results[(det_results.det_type == 'det_fp') | (det_results.det_type == 'det_tp')].sort_values('pred_score', ascending=False)
    # calculate the detection FROC
    fp, recall, key_recall, avg_recall = froc(det_results, num_gts)


    eval_results = {
        "detection": {
            "fp": fp,
            "recall": recall,
            "key_recall": key_recall,
            "average_recall": avg_recall,
            "max_recall": max(recall),
            "average_fp_at_max_recall": max(fp),
        }
    }

    return eval_results

if __name__ == "__main__":

    test_df="./examples/400/fold_0_test_df.pickle"
    det_results=pd.read_pickle(test_df)
    eval_results = evaluate(det_results)

    # detection metrics
    print("\nDetection metrics")
    print("=" * 64)
    print("Recall at key FP")
    print(pd.DataFrame(np.array(eval_results["detection"]["key_recall"])\
        .reshape(1, -1), index=["Recall"],
        columns=[f"FP={str(x)}" for x in DEFAULT_KEY_FP]))
    print("Average recall: {:.4f}".format(
        eval_results["detection"]["average_recall"]))
    print("Maximum recall: {:.4f}".format(
        eval_results["detection"]["max_recall"]
    ))
    print("Average FP per scan at maximum recall: {:.4f}".format(
        eval_results["detection"]["average_fp_at_max_recall"]
    ))
    # plot/print FROC curve
    print("FPR, Recall in FROC")
    for fp, recall in zip(reversed(eval_results["detection"]["fp"]),
            reversed(eval_results["detection"]["recall"])):
        print(f"({fp:.8f}, {recall:.8f})")
    plot_froc(eval_results["detection"]["fp"],
        eval_results["detection"]["recall"])
