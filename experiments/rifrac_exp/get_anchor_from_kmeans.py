import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


# 参考博客： https://zhuanlan.zhihu.com/p/109968578

class AnchorKmeans(object):
    """
    K-means clustering on bounding boxes to generate anchors
    """
    def __init__(self, k, max_iter=300, random_seed=None):
        self.k = k
        self.max_iter = max_iter
        self.random_seed = random_seed
        self.n_iter = 0
        self.anchors_ = None
        self.labels_ = None
        self.ious_ = None

    def fit(self, boxes):
        """
        Run K-means cluster on input boxes.
        :param boxes: 2-d array, shape(n, 2), form as (w, h)
        :return: None
        """
        assert self.k < len(boxes), "K must be less than the number of data."

        # If the current number of iterations is greater than 0, then reset
        if self.n_iter > 0:
            self.n_iter = 0

        np.random.seed(self.random_seed)
        n = boxes.shape[0]

        # Initialize K cluster centers (i.e., K anchors)
        self.anchors_ = boxes[np.random.choice(n, self.k, replace=True)]

        self.labels_ = np.zeros((n,))

        while True:
            self.n_iter += 1

            # If the current number of iterations is greater than max number of iterations , then break
            if self.n_iter > self.max_iter:
                break

            self.ious_ = self.iou(boxes, self.anchors_)
            distances = 1 - self.ious_
            cur_labels = np.argmin(distances, axis=1)

            # If anchors not change any more, then break
            if (cur_labels == self.labels_).all():
                break

            # Update K anchors
            for i in range(self.k):
                self.anchors_[i] = np.mean(boxes[cur_labels == i], axis=0)

            self.labels_ = cur_labels

    @staticmethod
    def iou(boxes, anchors):
        """
        Calculate the IOU between boxes and anchors.
        :param boxes: 2-d array, shape(n, 2)
        :param anchors: 2-d array, shape(k, 2)
        :return: 2-d array, shape(n, k)
        """
        # Calculate the intersection,
        # the new dimension are added to construct shape (n, 1) and shape (1, k),
        # so we can get (n, k) shape result by numpy broadcast
        z_min = np.minimum(boxes[:, 0, np.newaxis], anchors[np.newaxis, :, 0])
        w_min = np.minimum(boxes[:, 1, np.newaxis], anchors[np.newaxis, :, 1])
        h_min = np.minimum(boxes[:, 2, np.newaxis], anchors[np.newaxis, :, 2])
        inter = w_min * h_min * z_min

        # Calculate the union
        box_area = boxes[:, 0] * boxes[:, 1] * boxes[:,2]
        anchor_area = anchors[:, 0] * anchors[:, 1]*anchors[:,2]
        union = box_area[:, np.newaxis] + anchor_area[np.newaxis]

        return inter / (union - inter)

    def avg_iou(self):
        """
        Calculate the average IOU with closest anchor.
        :return: None
        """
        return np.mean(self.ious_[np.arange(len(self.labels_)), self.labels_])

if __name__=="__main__":
    # 准备数据
    boxes_dir="./rois_features/rois_feature.pickle"
    boxes_list=pd.read_pickle(boxes_dir)
    boxes=[]
    for item in boxes_list:
        for box in item['bbox']: #zmin,xmin,ymin,zmax,xmax,ymax
            boxes.append(np.array(box))
    boxes=np.array(boxes)
    # patch的图像大小，w,h
    z,w,h=64,128,128
    # 将boxes长宽高归一化
    boxes_norm=np.zeros((boxes.shape[0],3))
    boxes_norm[:,0]=(boxes[:,3]-boxes[:,0])/z
    boxes_norm[:,1]=(boxes[:,4]-boxes[:,1])/w
    boxes_norm[:,2]=(boxes[:,5]-boxes[:,2])/h
    # 选择（x,y）
    print(boxes_norm.shape)

    k=4
    print("[INFO] Initialize model")
    model = AnchorKmeans(k)

    print("[INFO] Training...")
    model.fit(boxes_norm)

    anchors = model.anchors_*np.array([64,128,128])
    print("[INFO] The results anchors:\n{}".format(anchors))

    k_list,IoU=[],[]
    for k in range(2, 11):
        model = AnchorKmeans(k, random_seed=333)
        model.fit(boxes_norm)
        avg_iou = model.avg_iou()
        print("K = {}, Avg IOU = {:.4f}".format(k, avg_iou))
        k_list.append(k)
        IoU.append(avg_iou)

    plt.plot(k_list,IoU)
    plt.show()