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

"""
Retina Net. According to https://arxiv.org/abs/1708.02002
Retina U-Net. According to https://arxiv.org/abs/1811.08661
"""

import utils.model_utils as mutils
import utils.exp_utils as utils
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils

sys.path.append('../')
from custom_extensions.nms import nms
from custom_extensions.roi_align import roi_align


"""
function:FP reduce
"""
class Fp_reduce(nn.Module):
    """
        Head network for classification and bounding box refinement. Performs RoiAlign, processes resulting features through a
        shared convolutional base and finally branches off the classifier- and regression head.
        """
    def __init__(self, cf, conv):
        super(Fp_reduce, self).__init__()
        self.dim = conv.dim
        self.in_channels = cf.end_filts
        self.pool_size = cf.pool_size
        self.pyramid_levels = [0]
        # instance_norm does not work with spatial dims (1, 1, (1))
        norm = cf.norm if cf.norm != 'instance_norm' else None

        self.conv1 = conv(cf.end_filts, cf.end_filts * 4, ks=self.pool_size, stride=1, norm=norm, relu=cf.relu)
        self.conv2 = conv(cf.end_filts * 4, cf.end_filts * 4, ks=1, stride=1, norm=norm, relu=cf.relu)
        self.linear_class = nn.Linear(cf.end_filts * 4, cf.head_classes)
        # self.linear_bbox = nn.Linear(cf.end_filts * 4, cf.head_classes * 2 * self.dim)

    def forward(self, x, rois):
        """
        :param x: input feature maps (b, in_channels, y, x, (z))
        :param rois: normalized box coordinates as proposed by the RPN to be forwarded through
        the second stage (n_proposals, (y1, x1, y2, x2, (z1), (z2), batch_ix). Proposals of all batch elements
        have been merged to one vector, while the origin info has been stored for re-allocation.
        :return: mrcnn_class_logits (n_proposals, n_head_classes)
        :return: mrcnn_bbox (n_proposals, n_head_classes, 2 * dim) predicted corrections to be applied to proposals for refinement.
        """
        x = fp_roi_align(x, rois, self.pool_size, self.pyramid_levels, self.dim)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1, self.in_channels * 4)
        mrcnn_class_logits = self.linear_class(x)
        # mrcnn_bbox = self.linear_bbox(x)
        # mrcnn_bbox = mrcnn_bbox.view(mrcnn_bbox.size()[0], -1, self.dim * 2)

        # return [mrcnn_class_logits, mrcnn_bbox]
        return mrcnn_class_logits

############################################################
#  Network Heads
############################################################

class Classifier(nn.Module):


    def __init__(self, cf, conv):
        """
        Builds the classifier sub-network.
        """
        super(Classifier, self).__init__()
        self.dim = conv.dim
        self.n_classes = cf.head_classes
        n_input_channels = cf.end_filts
        n_features = cf.n_rpn_features
        n_output_channels = cf.n_anchors_per_pos * cf.head_classes
        anchor_stride = cf.rpn_anchor_stride

        self.conv_1 = conv(n_input_channels, n_features, ks=3, stride=anchor_stride, pad=1, relu=cf.relu)
        self.conv_2 = conv(n_features, n_features, ks=3, stride=anchor_stride, pad=1, relu=cf.relu)
        self.conv_3 = conv(n_features, n_features, ks=3, stride=anchor_stride, pad=1, relu=cf.relu)
        self.conv_4 = conv(n_features, n_features, ks=3, stride=anchor_stride, pad=1, relu=cf.relu)
        self.conv_final = conv(n_features, n_output_channels, ks=3, stride=anchor_stride, pad=1, relu=None)


    def forward(self, x):
        """
        :param x: input feature map (b, in_c, y, x, (z))
        :return: class_logits (b, n_anchors, n_classes)
        """
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        class_logits = self.conv_final(x)

        axes = (0, 2, 3, 1) if self.dim == 2 else (0, 2, 3, 4, 1)
        class_logits = class_logits.permute(*axes)
        class_logits = class_logits.contiguous()
        class_logits = class_logits.view(x.size()[0], -1, self.n_classes)

        return [class_logits]



class BBRegressor(nn.Module):


    def __init__(self, cf, conv):
        """
        Builds the bb-regression sub-network.
        """
        super(BBRegressor, self).__init__()
        self.dim = conv.dim
        n_input_channels = cf.end_filts
        n_features = cf.n_rpn_features
        n_output_channels = cf.n_anchors_per_pos * self.dim * 2
        anchor_stride = cf.rpn_anchor_stride

        self.conv_1 = conv(n_input_channels, n_features, ks=3, stride=anchor_stride, pad=1, relu=cf.relu)
        self.conv_2 = conv(n_features, n_features, ks=3, stride=anchor_stride, pad=1, relu=cf.relu)
        self.conv_3 = conv(n_features, n_features, ks=3, stride=anchor_stride, pad=1, relu=cf.relu)
        self.conv_4 = conv(n_features, n_features, ks=3, stride=anchor_stride, pad=1, relu=cf.relu)
        self.conv_final = conv(n_features, n_output_channels, ks=3, stride=anchor_stride,
                               pad=1, relu=None)

    def forward(self, x):
        """
        :param x: input feature map (b, in_c, y, x, (z))
        :return: bb_logits (b, n_anchors, dim * 2)
        """
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        bb_logits = self.conv_final(x)

        axes = (0, 2, 3, 1) if self.dim == 2 else (0, 2, 3, 4, 1)
        bb_logits = bb_logits.permute(*axes)
        bb_logits = bb_logits.contiguous()
        bb_logits = bb_logits.view(x.size()[0], -1, self.dim * 2)

        return [bb_logits]


def refine_proposals(rpn_pred_probs, rpn_pred_deltas, proposal_count, batch_anchors, cf):
    """
    Receives anchor scores and selects a subset to pass as proposals
    to the second stage. Filtering is done based on anchor scores and
    non-max suppression to remove overlaps. It also applies bounding
    box refinment details to anchors.
    :param rpn_pred_probs: (b, n_anchors, 2)
    :param rpn_pred_deltas: (b, n_anchors, (y, x, (z), log(h), log(w), (log(d))))
    :return: batch_normalized_props: Proposals in normalized coordinates (b, proposal_count, (y1, x1, y2, x2, (z1), (z2), score))
    :return: batch_out_proposals: Box coords + RPN foreground scores
    for monitoring/plotting (b, proposal_count, (y1, x1, y2, x2, (z1), (z2), score))
    """
    std_dev = torch.from_numpy(cf.rpn_bbox_std_dev[None]).float().cuda()
    norm = torch.from_numpy(cf.scale).float().cuda()
    anchors = batch_anchors.clone()

    batch_scores = rpn_pred_probs[:, :, 1]
    # norm deltas
    batch_deltas = rpn_pred_deltas * std_dev
    batch_normalized_props = []
    batch_out_proposals = []

    # loop over batch dimension.
    for ix in range(batch_scores.shape[0]):

        scores = batch_scores[ix]
        deltas = batch_deltas[ix]

        # improve performance by trimming to top anchors by score
        # and doing the rest on the smaller subset.
        pre_nms_limit = min(cf.pre_nms_limit, anchors.size()[0])
        scores, order = scores.sort(descending=True)
        order = order[:pre_nms_limit]
        scores = scores[:pre_nms_limit]
        deltas = deltas[order, :]

        # apply deltas to anchors to get refined anchors and filter with non-maximum suppression.
        if batch_deltas.shape[-1] == 4:
            boxes = mutils.apply_box_deltas_2D(anchors[order, :], deltas)
            boxes = mutils.clip_boxes_2D(boxes, cf.window)
        else:
            boxes = mutils.apply_box_deltas_3D(anchors[order, :], deltas)
            boxes = mutils.clip_boxes_3D(boxes, cf.window)
        # boxes are y1,x1,y2,x2, torchvision-nms requires x1,y1,x2,y2, but consistent swap x<->y is irrelevant.
        keep = nms.nms(boxes, scores, cf.rpn_nms_threshold)


        keep = keep[:proposal_count]
        boxes = boxes[keep, :]
        rpn_scores = scores[keep][:, None]

        # pad missing boxes with 0.
        if boxes.shape[0] < proposal_count:
            n_pad_boxes = proposal_count - boxes.shape[0]
            zeros = torch.zeros([n_pad_boxes, boxes.shape[1]]).cuda()
            boxes = torch.cat([boxes, zeros], dim=0)
            zeros = torch.zeros([n_pad_boxes, rpn_scores.shape[1]]).cuda()
            rpn_scores = torch.cat([rpn_scores, zeros], dim=0)

        # concat box and score info for monitoring/plotting.
        batch_out_proposals.append(torch.cat((boxes, rpn_scores), 1).cpu().data.numpy())
        # normalize dimensions to range of 0 to 1.
        normalized_boxes = boxes / norm
        assert torch.all(normalized_boxes <= 1), "normalized box coords >1 found"

        # add again batch dimension
        batch_normalized_props.append(normalized_boxes.unsqueeze(0))

    batch_normalized_props = torch.cat(batch_normalized_props)
    batch_out_proposals = np.array(batch_out_proposals)

    return batch_normalized_props, batch_out_proposals

# def fp_roi_align(feature_maps, rois, pool_size, pyramid_levels, dim):
#     """
#     Implements ROI Pooling on multiple levels of the feature pyramid.
#     :param feature_maps: list of feature maps, each of shape (b, c, y, x , (z))
#     :param rois: proposals (normalized coords.) as returned by RPN. contain info about original batch element allocation.
#     (n_proposals, (y1, x1, y2, x2, (z1), (z2), batch_ixs)
#     :param pool_size: list of poolsizes in dims: [x, y, (z)]
#     :param pyramid_levels: list. [0, 1, 2, ...]
#     :return: pooled: pooled feature map rois (n_proposals, c, poolsize_y, poolsize_x, (poolsize_z))
#
#     Output:
#     Pooled regions in the shape: [num_boxes, height, width, channels].
#     The width and height are those specific in the pool_shape in the layer
#     constructor.
#     """
#     boxes = rois[:, :dim*2]
#     batch_ixs = rois[:, dim*2]
#
#     # Assign each ROI to a level in the pyramid based on the ROI area.
#     if dim == 2:
#         y1, x1, y2, x2 = boxes.chunk(4, dim=1)
#     else:
#         y1, x1, y2, x2, z1, z2 = boxes.chunk(6, dim=1)
#
#     h = y2 - y1
#     w = x2 - x1
#
#     # Equation 1 in https://arxiv.org/abs/1612.03144. Account for
#     # the fact that our coordinates are normalized here.
#     # divide sqrt(h*w) by 1 instead image_area.
#     # roi_level = (4 + torch.log2(torch.sqrt(h*w))).round().int().clamp(pyramid_levels[0], pyramid_levels[-1])
#     # if Pyramid contains additional level P6, adapt the roi_level assignment accordingly.
#     # if len(pyramid_levels) == 5:
#     #     roi_level[h*w > 0.65] = 5
#
#     # Loop through levels and apply ROI pooling to each.
#     pooled = []
#     # box_to_level = []
#     fmap_shapes = [f.shape for f in feature_maps]
#     for level_ix, level in enumerate(pyramid_levels):
#         boxes=boxes.detach()
#         if len(pool_size) == 2:
#             # remap to feature map coordinate system
#             y_exp, x_exp = fmap_shapes[level_ix][2:]  # exp = expansion
#             boxes.mul_(torch.tensor([y_exp, x_exp, y_exp, x_exp], dtype=torch.float32).cuda())
#             pooled_features = roi_align.roi_align_2d(feature_maps[level_ix],
#                                                      torch.cat((batch_ixs.int().unsqueeze(1).float(), boxes), dim=1),
#                                                      pool_size)
#         else:
#             y_exp, x_exp, z_exp = fmap_shapes[level_ix][2:]
#             boxes.mul_(torch.tensor([y_exp, x_exp, y_exp, x_exp, z_exp, z_exp], dtype=torch.float32).cuda())
#             # boxes_new=torch.mul(boxes,torch.tensor([y_exp, x_exp, y_exp, x_exp, z_exp, z_exp], dtype=torch.float32).cuda())
#             pooled_features = roi_align.roi_align_3d(feature_maps[level_ix],
#                                                      torch.cat((batch_ixs.int().unsqueeze(1).float(), boxes), dim=1),
#                                                      pool_size)
#         pooled.append(pooled_features)
#
#
#     # Pack pooled features into one tensor
#     pooled = torch.cat(pooled, dim=0)
#
#     # Pack box_to_level mapping into one array and add another
#     # column representing the order of pooled boxes
#     # box_to_level = torch.cat(box_to_level, dim=0)
#
#     # Rearrange pooled features to match the order of the original boxes
#     # _, box_to_level = torch.sort(box_to_level)
#     # pooled = pooled[box_to_level, :, :]
#
#     return pooled
#
def fp_roi_align(feature_maps, rois, pool_size, pyramid_levels, dim):
    """
    Implements ROI Pooling on multiple levels of the feature pyramid.
    :param feature_maps: list of feature maps, each of shape (b, c, y, x , (z))
    :param rois: proposals (normalized coords.) as returned by RPN. contain info about original batch element allocation.
    (n_proposals, (y1, x1, y2, x2, (z1), (z2), batch_ixs)
    :param pool_size: list of poolsizes in dims: [x, y, (z)]
    :param pyramid_levels: list. [0, 1, 2, ...]
    :return: pooled: pooled feature map rois (n_proposals, c, poolsize_y, poolsize_x, (poolsize_z))

    Output:
    Pooled regions in the shape: [num_boxes, height, width, channels].
    The width and height are those specific in the pool_shape in the layer
    constructor.
    """
    boxes = rois[:, :dim*2]
    batch_ixs = rois[:, dim*2]

    # Assign each ROI to a level in the pyramid based on the ROI area.
    if dim == 2:
        y1, x1, y2, x2 = boxes.chunk(4, dim=1)
    else:
        y1, x1, y2, x2, z1, z2 = boxes.chunk(6, dim=1)

    h = y2 - y1
    w = x2 - x1

    # Equation 1 in https://arxiv.org/abs/1612.03144. Account for
    # the fact that our coordinates are normalized here.
    # divide sqrt(h*w) by 1 instead image_area.
    roi_level = (4 + torch.log2(torch.sqrt(h*w))).round().int().clamp(pyramid_levels[0], pyramid_levels[-1])
    # if Pyramid contains additional level P6, adapt the roi_level assignment accordingly.
    if len(pyramid_levels) == 5:
        roi_level[h*w > 0.65] = 5

    # Loop through levels and apply ROI pooling to each.
    pooled = []
    box_to_level = []
    fmap_shapes = [f.shape for f in feature_maps]
    for level_ix, level in enumerate(pyramid_levels):
        ix = roi_level == level
        if not ix.any():
            continue
        ix = torch.nonzero(ix)[:, 0]
        level_boxes = boxes[ix, :]
        # re-assign rois to feature map of original batch element.
        ind = batch_ixs[ix].int()

        # Keep track of which box is mapped to which level
        box_to_level.append(ix)

        # Stop gradient propogation to ROI proposals
        level_boxes = level_boxes.detach()
        if len(pool_size) == 2:
            # remap to feature map coordinate system
            y_exp, x_exp = fmap_shapes[level_ix][2:]  # exp = expansion
            level_boxes.mul_(torch.tensor([y_exp, x_exp, y_exp, x_exp], dtype=torch.float32).cuda())
            pooled_features = roi_align.roi_align_2d(feature_maps[level_ix],
                                                     torch.cat((ind.unsqueeze(1).float(), level_boxes), dim=1),
                                                     pool_size)
        else:
            y_exp, x_exp, z_exp = fmap_shapes[level_ix][2:]
            level_boxes.mul_(torch.tensor([y_exp, x_exp, y_exp, x_exp, z_exp, z_exp], dtype=torch.float32).cuda())
            pooled_features = roi_align.roi_align_3d(feature_maps[level_ix],
                                                     torch.cat((ind.unsqueeze(1).float(), level_boxes), dim=1),
                                                     pool_size)
        pooled.append(pooled_features)


    # Pack pooled features into one tensor
    pooled = torch.cat(pooled, dim=0)

    # Pack box_to_level mapping into one array and add another
    # column representing the order of pooled boxes
    box_to_level = torch.cat(box_to_level, dim=0)

    # Rearrange pooled features to match the order of the original boxes
    _, box_to_level = torch.sort(box_to_level)
    pooled = pooled[box_to_level, :, :]

    return pooled


def pyramid_roi_align(feature_maps, rois, pool_size, pyramid_levels, dim):
    """
    Implements ROI Pooling on multiple levels of the feature pyramid.
    :param feature_maps: list of feature maps, each of shape (b, c, y, x , (z))
    :param rois: proposals (normalized coords.) as returned by RPN. contain info about original batch element allocation.
    (n_proposals, (y1, x1, y2, x2, (z1), (z2), batch_ixs)
    :param pool_size: list of poolsizes in dims: [x, y, (z)]
    :param pyramid_levels: list. [0, 1, 2, ...]
    :return: pooled: pooled feature map rois (n_proposals, c, poolsize_y, poolsize_x, (poolsize_z))

    Output:
    Pooled regions in the shape: [num_boxes, height, width, channels].
    The width and height are those specific in the pool_shape in the layer
    constructor.
    """
    boxes = rois[:, :dim*2]
    batch_ixs = rois[:, dim*2]

    # Assign each ROI to a level in the pyramid based on the ROI area.
    if dim == 2:
        y1, x1, y2, x2 = boxes.chunk(4, dim=1)
    else:
        y1, x1, y2, x2, z1, z2 = boxes.chunk(6, dim=1)

    h = y2 - y1
    w = x2 - x1

    # Equation 1 in https://arxiv.org/abs/1612.03144. Account for
    # the fact that our coordinates are normalized here.
    # divide sqrt(h*w) by 1 instead image_area.
    roi_level = (4 + torch.log2(torch.sqrt(h*w))).round().int().clamp(pyramid_levels[0], pyramid_levels[-1])
    # if Pyramid contains additional level P6, adapt the roi_level assignment accordingly.
    if len(pyramid_levels) == 5:
        roi_level[h*w > 0.65] = 5

    # Loop through levels and apply ROI pooling to each.
    pooled = []
    box_to_level = []
    fmap_shapes = [f.shape for f in feature_maps]
    for level_ix, level in enumerate(pyramid_levels):
        ix = roi_level == level
        if not ix.any():
            continue
        ix = torch.nonzero(ix)[:, 0]
        level_boxes = boxes[ix, :]
        # re-assign rois to feature map of original batch element.
        ind = batch_ixs[ix].int()

        # Keep track of which box is mapped to which level
        box_to_level.append(ix)

        # Stop gradient propogation to ROI proposals
        level_boxes = level_boxes.detach()
        if len(pool_size) == 2:
            # remap to feature map coordinate system
            y_exp, x_exp = fmap_shapes[level_ix][2:]  # exp = expansion
            level_boxes.mul_(torch.tensor([y_exp, x_exp, y_exp, x_exp], dtype=torch.float32).cuda())
            pooled_features = roi_align.roi_align_2d(feature_maps[level_ix],
                                                     torch.cat((ind.unsqueeze(1).float(), level_boxes), dim=1),
                                                     pool_size)
        else:
            y_exp, x_exp, z_exp = fmap_shapes[level_ix][2:]
            level_boxes.mul_(torch.tensor([y_exp, x_exp, y_exp, x_exp, z_exp, z_exp], dtype=torch.float32).cuda())
            pooled_features = roi_align.roi_align_3d(feature_maps[level_ix],
                                                     torch.cat((ind.unsqueeze(1).float(), level_boxes), dim=1),
                                                     pool_size)
        pooled.append(pooled_features)


    # Pack pooled features into one tensor
    pooled = torch.cat(pooled, dim=0)

    # Pack box_to_level mapping into one array and add another
    # column representing the order of pooled boxes
    box_to_level = torch.cat(box_to_level, dim=0)

    # Rearrange pooled features to match the order of the original boxes
    _, box_to_level = torch.sort(box_to_level)
    pooled = pooled[box_to_level, :, :]

    return pooled

def detection_target_layer(batch_proposals, batch_mrcnn_class_scores, batch_gt_class_ids, batch_gt_boxes, batch_gt_masks, cf):
    """
    Subsamples proposals for mrcnn losses and generates targets. Sampling is done per batch element, seems to have positive
    effects on training, as opposed to sampling over entire batch. Negatives are sampled via stochastic-hard-example-mining
    (SHEM), where a number of negative proposals are drawn from larger pool of highest scoring proposals for stochasticity.
    Scoring is obtained here as the max over all foreground probabilities as returned by mrcnn_classifier (worked better than
    loss-based class balancing methods like "online-hard-example-mining" or "focal loss".)

    :param batch_proposals: (n_proposals, (y1, x1, y2, x2, (z1), (z2), batch_ixs).
    boxes as proposed by RPN. n_proposals here is determined by batch_size * POST_NMS_ROIS.
    :param batch_mrcnn_class_scores: (n_proposals, n_classes)
    :param batch_gt_class_ids: list over batch elements. Each element is a list over the corresponding roi target labels.
    :param batch_gt_boxes: list over batch elements. Each element is a list over the corresponding roi target coordinates.
    :param batch_gt_masks: list over batch elements. Each element is binary mask of shape (n_gt_rois, y, x, (z), c)
    :return: sample_indices: (n_sampled_rois) indices of sampled proposals to be used for loss functions.
    :return: target_class_ids: (n_sampled_rois)containing target class labels of sampled proposals.
    :return: target_deltas: (n_sampled_rois, 2 * dim) containing target deltas of sampled proposals for box refinement.
    :return: target_masks: (n_sampled_rois, y, x, (z)) containing target masks of sampled proposals.
    """
    # normalization of target coordinates
    if cf.dim == 2:
        h, w = cf.patch_size
        scale = torch.from_numpy(np.array([h, w, h, w])).float().cuda()
    else:
        h, w, z = cf.patch_size
        scale = torch.from_numpy(np.array([h, w, h, w, z, z])).float().cuda()

    positive_count = 0
    negative_count = 0
    sample_positive_indices = []
    sample_negative_indices = []
    sample_deltas = []
    sample_masks = []
    sample_class_ids = []

    std_dev = torch.from_numpy(cf.bbox_std_dev).float().cuda()

    # loop over batch and get positive and negative sample rois.
    for b in range(len(batch_gt_class_ids)):

        gt_class_ids = torch.from_numpy(batch_gt_class_ids[b]).int().cuda()
        gt_masks = torch.from_numpy(batch_gt_masks[b]).float().cuda()
        if np.any(batch_gt_class_ids[b] > 0):  # skip roi selection for no gt images.
            gt_boxes = torch.from_numpy(batch_gt_boxes[b]).float().cuda() / scale
        else:
            gt_boxes = torch.FloatTensor().cuda()

        # get proposals and indices of current batch element.
        proposals = batch_proposals[batch_proposals[:, -1] == b][:, :-1]
        batch_element_indices = torch.nonzero(batch_proposals[:, -1] == b).squeeze(1)

        # Compute overlaps matrix [proposals, gt_boxes]
        if 0 not in gt_boxes.size():
            if gt_boxes.shape[1] == 4:
                assert cf.dim == 2, "gt_boxes shape {} doesnt match cf.dim{}".format(gt_boxes.shape, cf.dim)
                overlaps = mutils.bbox_overlaps_2D(proposals, gt_boxes)
            else:
                assert cf.dim == 3, "gt_boxes shape {} doesnt match cf.dim{}".format(gt_boxes.shape, cf.dim)
                overlaps = mutils.bbox_overlaps_3D(proposals, gt_boxes)

            # Determine postive and negative ROIs
            roi_iou_max = torch.max(overlaps, dim=1)[0]
            # 1. Positive ROIs are those with >= 0.5 IoU with a GT box
            positive_roi_bool = roi_iou_max >= (0.5 if cf.dim == 2 else 0.3)
            # 2. Negative ROIs are those with < 0.1 with every GT box.
            negative_roi_bool = roi_iou_max < (0.1 if cf.dim == 2 else 0.01)
        else:
            positive_roi_bool = torch.FloatTensor().cuda()
            negative_roi_bool = torch.from_numpy(np.array([1]*proposals.shape[0])).cuda()

        # Sample Positive ROIs
        if 0 not in torch.nonzero(positive_roi_bool).size():
            positive_indices = torch.nonzero(positive_roi_bool).squeeze(1)
            positive_samples = int(cf.train_rois_per_image * cf.roi_positive_ratio)
            rand_idx = torch.randperm(positive_indices.size()[0])
            rand_idx = rand_idx[:positive_samples].cuda()
            positive_indices = positive_indices[rand_idx]
            positive_samples = positive_indices.size()[0]
            positive_rois = proposals[positive_indices, :]
            # Assign positive ROIs to GT boxes.
            positive_overlaps = overlaps[positive_indices, :]
            roi_gt_box_assignment = torch.max(positive_overlaps, dim=1)[1]
            roi_gt_boxes = gt_boxes[roi_gt_box_assignment, :]
            roi_gt_class_ids = gt_class_ids[roi_gt_box_assignment]

            # Compute bbox refinement targets for positive ROIs
            deltas = mutils.box_refinement(positive_rois, roi_gt_boxes)
            deltas /= std_dev

            # Assign positive ROIs to GT masks
            roi_masks = gt_masks[roi_gt_box_assignment]
            assert roi_masks.shape[1] == 1, "desired to have more than one channel in gt masks?"

            # Compute mask targets
            boxes = positive_rois
            box_ids = torch.arange(roi_masks.shape[0]).cuda().unsqueeze(1).float()
            if len(cf.mask_shape) == 2:
                # need to remap normalized box coordinates to unnormalized mask coordinates.
                y_exp, x_exp = roi_masks.shape[2:]  # exp = expansion
                boxes.mul_(torch.tensor([y_exp, x_exp, y_exp, x_exp], dtype=torch.float32).cuda())
                masks = roi_align.roi_align_2d(roi_masks, torch.cat((box_ids, boxes), dim=1), cf.mask_shape)
            else:
                y_exp, x_exp, z_exp = roi_masks.shape[2:]  # exp = expansion
                # boxes.mul_(torch.tensor([y_exp, x_exp, y_exp, x_exp, z_exp, z_exp], dtype=torch.float32).cuda())
                boxes_new=torch.mul(boxes,torch.tensor([y_exp, x_exp, y_exp, x_exp, z_exp, z_exp], dtype=torch.float32).cuda())
                masks = roi_align.roi_align_3d(roi_masks, torch.cat((box_ids, boxes_new), dim=1), cf.mask_shape)
            masks = masks.squeeze(1)
            # Threshold mask pixels at 0.5 to have GT masks be 0 or 1 to use with
            # binary cross entropy loss.
            masks = torch.round(masks)

            sample_positive_indices.append(batch_element_indices[positive_indices])
            sample_deltas.append(deltas)
            sample_masks.append(masks)
            sample_class_ids.append(roi_gt_class_ids)
            positive_count += positive_samples
        else:
            positive_samples = 0

        # Negative ROIs. Add enough to maintain positive:negative ratio, but at least 1. Sample via SHEM.
        if 0 not in torch.nonzero(negative_roi_bool).size():
            negative_indices = torch.nonzero(negative_roi_bool).squeeze(1)
            r = 1.0 / cf.roi_positive_ratio
            b_neg_count = np.max((int(r * positive_samples - positive_samples), 1))
            roi_probs_neg = batch_mrcnn_class_scores[batch_element_indices[negative_indices]]
            raw_sampled_indices = mutils.shem(roi_probs_neg, b_neg_count, cf.shem_poolsize)
            sample_negative_indices.append(batch_element_indices[negative_indices[raw_sampled_indices]])
            negative_count += raw_sampled_indices.size()[0]

    if len(sample_positive_indices) > 0:
        target_deltas = torch.cat(sample_deltas)
        target_masks = torch.cat(sample_masks)
        target_class_ids = torch.cat(sample_class_ids)

    # print("\n")
    # print(batch_proposals.shape[0],len(batch_gt_class_ids[0]),positive_count,negative_count)
    # Pad target information with zeros for negative ROIs.
    if positive_count > 0 and negative_count > 0:
        sample_indices = torch.cat((torch.cat(sample_positive_indices), torch.cat(sample_negative_indices)), dim=0)
        zeros = torch.zeros(negative_count).int().cuda()
        target_class_ids = torch.cat([target_class_ids, zeros], dim=0)
        zeros = torch.zeros(negative_count, cf.dim * 2).cuda()
        target_deltas = torch.cat([target_deltas, zeros], dim=0)
        zeros = torch.zeros(negative_count, *cf.mask_shape).cuda()
        target_masks = torch.cat([target_masks, zeros], dim=0)
    elif positive_count > 0:
        sample_indices = torch.cat(sample_positive_indices)
    elif negative_count > 0:
        sample_indices = torch.cat(sample_negative_indices)
        zeros = torch.zeros(negative_count).int().cuda()
        target_class_ids = zeros
        zeros = torch.zeros(negative_count, cf.dim * 2).cuda()
        target_deltas = zeros
        zeros = torch.zeros(negative_count, *cf.mask_shape).cuda()
        target_masks = zeros
    else:
        sample_indices = torch.LongTensor().cuda()
        target_class_ids = torch.IntTensor().cuda()
        target_deltas = torch.FloatTensor().cuda()
        target_masks = torch.FloatTensor().cuda()

    return sample_indices, target_class_ids, target_deltas, target_masks

############################################################
#  Loss Functions
############################################################
def compute_fp_reduce_class_loss(target_class_ids, pred_class_logits):
    """
    :param target_class_ids: (n_sampled_rois) batch dimension was merged into roi dimension.
    :param pred_class_logits: (n_sampled_rois, n_classes)
    :return: loss: torch 1D tensor.
    """
    if 0 not in target_class_ids.size():
        loss = F.cross_entropy(pred_class_logits, target_class_ids.long())
    else:
        loss = torch.FloatTensor([0.]).cuda()

    return loss

def compute_mrcnn_mask_loss(target_masks, pred_masks, target_class_ids):
    """
    :param target_masks: (n_sampled_rois, y, x, (z)) A float32 tensor of values 0 or 1. Uses zero padding to fill array.
    :param pred_masks: (n_sampled_rois, n_classes, y, x, (z)) float32 tensor with values between [0, 1].
    :param target_class_ids: (n_sampled_rois)
    :return: loss: torch 1D tensor.
    """
    if 0 not in torch.nonzero(target_class_ids > 0).size():
        # Only positive ROIs contribute to the loss. And only
        # the class specific mask of each ROI.
        positive_ix = torch.nonzero(target_class_ids > 0)[:, 0]
        positive_class_ids = target_class_ids[positive_ix].long()
        y_true = target_masks[positive_ix, :, :].detach()
        y_pred = pred_masks[positive_ix, positive_class_ids, :, :]
        loss = F.binary_cross_entropy(y_pred, y_true)
    else:
        loss = torch.FloatTensor([0]).cuda()

    return loss

def compute_class_loss(anchor_matches, class_pred_logits, shem_poolsize=20):
    """
    :param anchor_matches: (n_anchors). [-1, 0, class_id] for negative, neutral, and positive matched anchors.
    :param class_pred_logits: (n_anchors, n_classes). logits from classifier sub-network.
    :param shem_poolsize: int. factor of top-k candidates to draw from per negative sample (online-hard-example-mining).
    :return: loss: torch tensor.
    :return: np_neg_ix: 1D array containing indices of the neg_roi_logits, which have been sampled for training.
    """
    # Positive and Negative anchors contribute to the loss,
    # but neutral anchors (match value = 0) don't.
    pos_indices = torch.nonzero(anchor_matches > 0)
    neg_indices = torch.nonzero(anchor_matches == -1)

    # get positive samples and calucalte loss.
    if 0 not in pos_indices.size():
        pos_indices = pos_indices.squeeze(1)
        roi_logits_pos = class_pred_logits[pos_indices]
        targets_pos = anchor_matches[pos_indices]
        pos_loss = F.cross_entropy(roi_logits_pos, targets_pos.long())
    else:
        pos_loss = torch.FloatTensor([0]).cuda()

    # get negative samples, such that the amount matches the number of positive samples, but at least 1.
    # get high scoring negatives by applying online-hard-example-mining.
    if 0 not in neg_indices.size():
        neg_indices = neg_indices.squeeze(1)
        roi_logits_neg = class_pred_logits[neg_indices]
        negative_count = np.max((1, pos_indices.size()[0]))
        roi_probs_neg = F.softmax(roi_logits_neg, dim=1)
        neg_ix = mutils.shem(roi_probs_neg, negative_count, shem_poolsize)
        neg_loss = F.cross_entropy(roi_logits_neg[neg_ix], torch.LongTensor([0] * neg_ix.shape[0]).cuda())
        # return the indices of negative samples, which contributed to the loss (for monitoring plots).
        np_neg_ix = neg_ix.cpu().data.numpy()
    else:
        neg_loss = torch.FloatTensor([0]).cuda()
        np_neg_ix = np.array([]).astype('int32')

    loss = (pos_loss + neg_loss) / 2
    return loss, np_neg_ix


def compute_bbox_loss(target_deltas, pred_deltas, anchor_matches):
    """
    :param target_deltas:   (b, n_positive_anchors, (dy, dx, (dz), log(dh), log(dw), (log(dd)))).
    Uses 0 padding to fill in unsed bbox deltas.
    :param pred_deltas: predicted deltas from bbox regression head. (b, n_anchors, (dy, dx, (dz), log(dh), log(dw), (log(dd))))
    :param anchor_matches: (n_anchors). [-1, 0, class_id] for negative, neutral, and positive matched anchors.
    :return: loss: torch 1D tensor.
    """
    if 0 not in torch.nonzero(anchor_matches > 0).size():

        indices = torch.nonzero(anchor_matches > 0).squeeze(1)
        # Pick bbox deltas that contribute to the loss
        pred_deltas = pred_deltas[indices]
        # Trim target bounding box deltas to the same length as pred_deltas.
        target_deltas = target_deltas[:pred_deltas.size()[0], :]
        # Smooth L1 loss
        loss = F.smooth_l1_loss(pred_deltas, target_deltas)
    else:
        loss = torch.FloatTensor([0]).cuda()

    return loss


def fp_get_detections(cf, batch_ixs, rois, deltas, scores):
    """
    Refine classified proposals (apply deltas to rpn rois), filter overlaps (nms) and return final detections.

    :param rois: (n_proposals, 2 * dim) normalized boxes as proposed by RPN. n_proposals = batch_size * POST_NMS_ROIS
    :param deltas: (n_proposals, n_classes, 2 * dim) box refinement deltas as predicted by mrcnn bbox regressor.
    :param batch_ixs: (n_proposals) batch element assignment info for re-allocation.
    :param scores: (n_proposals, n_classes) probabilities for all classes per roi as predicted by mrcnn classifier.
    :return: result: (n_final_detections, (y1, x1, y2, x2, (z1), (z2), batch_ix, pred_class_id, pred_score, *regression vector features))
    """
    # class IDs per ROI. Since scores of all classes are of interest (not just max class), all are kept at this point.
    class_ids = []
    fg_classes = cf.head_classes - 1
    # repeat vectors to fill in predictions for all foreground classes.
    for ii in range(1, fg_classes + 1):
        class_ids += [ii] * rois.shape[0]
    class_ids = torch.from_numpy(np.array(class_ids)).cuda()

    batch_ixs = batch_ixs.repeat(fg_classes)
    rois = rois.repeat(fg_classes, 1)
    deltas = deltas.repeat(fg_classes, 1, 1)
    scores = scores.repeat(fg_classes, 1)

    # get class-specific scores and  bounding box deltas
    idx = torch.arange(class_ids.size()[0]).long().cuda()
    # using idx instead of slice [:,] squashes first dimension.
    #len(class_ids)>scores.shape[1] --> probs is broadcasted by expansion from fg_classes-->len(class_ids)
    batch_ixs = batch_ixs[idx]
    deltas_specific = deltas[idx, class_ids]
    class_scores = scores[idx, class_ids]

    # apply bounding box deltas. re-scale to image coordinates.
    std_dev = torch.from_numpy(np.reshape(cf.rpn_bbox_std_dev, [1, cf.dim * 2])).float().cuda()
    scale = torch.from_numpy(cf.scale).float().cuda()
    refined_rois = mutils.apply_box_deltas_2D(rois, deltas_specific * std_dev) * scale if cf.dim == 2 else \
        mutils.apply_box_deltas_3D(rois, deltas_specific * std_dev) * scale

    # round and cast to int since we're dealing with pixels now
    refined_rois = mutils.clip_to_window(cf.window, refined_rois)
    refined_rois = torch.round(refined_rois)

    # filter out low confidence boxes
    keep = idx
    keep_bool = (class_scores >= cf.model_min_confidence)
    if not 0 in torch.nonzero(keep_bool).size():

        score_keep = torch.nonzero(keep_bool)[:, 0]
        pre_nms_class_ids = class_ids[score_keep]
        pre_nms_rois = refined_rois[score_keep]
        pre_nms_scores = class_scores[score_keep]
        pre_nms_batch_ixs = batch_ixs[score_keep]

        for j, b in enumerate(mutils.unique1d(pre_nms_batch_ixs)):

            bixs = torch.nonzero(pre_nms_batch_ixs == b)[:, 0]
            bix_class_ids = pre_nms_class_ids[bixs]
            bix_rois = pre_nms_rois[bixs]
            bix_scores = pre_nms_scores[bixs]

            for i, class_id in enumerate(mutils.unique1d(bix_class_ids)):

                ixs = torch.nonzero(bix_class_ids == class_id)[:, 0]
                # nms expects boxes sorted by score.
                ix_rois = bix_rois[ixs]
                ix_scores = bix_scores[ixs]
                ix_scores, order = ix_scores.sort(descending=True)
                ix_rois = ix_rois[order, :]

                class_keep = nms.nms(ix_rois, ix_scores, cf.detection_nms_threshold)

                # map indices back.
                class_keep = keep[score_keep[bixs[ixs[order[class_keep]]]]]
                # merge indices over classes for current batch element
                b_keep = class_keep if i == 0 else mutils.unique1d(torch.cat((b_keep, class_keep)))

            # only keep top-k boxes of current batch-element
            top_ids = class_scores[b_keep].sort(descending=True)[1][:cf.model_max_instances_per_batch_element]
            b_keep = b_keep[top_ids]

            # merge indices over batch elements.
            batch_keep = b_keep  if j == 0 else mutils.unique1d(torch.cat((batch_keep, b_keep)))

        keep = batch_keep

    else:
        keep = torch.tensor([0]).long().cuda()

    # arrange output
    output = [refined_rois[keep], batch_ixs[keep].unsqueeze(1)]
    output += [class_ids[keep].unsqueeze(1).float(), class_scores[keep].unsqueeze(1)]

    # arrange output
    output = [refined_rois, batch_ixs.unsqueeze(1)]
    output += [class_ids.unsqueeze(1).float(), class_scores.unsqueeze(1)]
    result = torch.cat(output, dim=1)
    # shape: (n_keeps, catted feats), catted feats: [0:dim*2] are box_coords, [dim*2] are batch_ics,
    # [dim*2+1] are class_ids, [dim*2+2] are scores, [dim*2+3:] are regression vector features (incl uncertainty)
    return result

def fp_refine_detections(cf, batch_ixs, rois, deltas, scores):
    """
    Refine classified proposals (apply deltas to rpn rois), filter overlaps (nms) and return final detections.

    :param rois: (n_proposals, 2 * dim) normalized boxes as proposed by RPN. n_proposals = batch_size * POST_NMS_ROIS
    :param deltas: (n_proposals, n_classes, 2 * dim) box refinement deltas as predicted by mrcnn bbox regressor.
    :param batch_ixs: (n_proposals) batch element assignment info for re-allocation.
    :param scores: (n_proposals, n_classes) probabilities for all classes per roi as predicted by mrcnn classifier.
    :return: result: (n_final_detections, (y1, x1, y2, x2, (z1), (z2), batch_ix, pred_class_id, pred_score, *regression vector features))
    """
    # class IDs per ROI. Since scores of all classes are of interest (not just max class), all are kept at this point.
    class_ids = []
    fg_classes = cf.head_classes - 1
    # repeat vectors to fill in predictions for all foreground classes.
    for ii in range(1, fg_classes + 1):
        class_ids += [ii] * rois.shape[0]
    class_ids = torch.from_numpy(np.array(class_ids)).cuda()

    batch_ixs = batch_ixs.repeat(fg_classes)
    rois = rois.repeat(fg_classes, 1)
    deltas = deltas.repeat(fg_classes, 1, 1)
    scores = scores.repeat(fg_classes, 1)

    # get class-specific scores and  bounding box deltas
    idx = torch.arange(class_ids.size()[0]).long().cuda()
    # using idx instead of slice [:,] squashes first dimension.
    #len(class_ids)>scores.shape[1] --> probs is broadcasted by expansion from fg_classes-->len(class_ids)
    batch_ixs = batch_ixs[idx]
    deltas_specific = deltas[idx, class_ids]
    class_scores = scores[idx, class_ids]

    # apply bounding box deltas. re-scale to image coordinates.
    std_dev = torch.from_numpy(np.reshape(cf.rpn_bbox_std_dev, [1, cf.dim * 2])).float().cuda()
    scale = torch.from_numpy(cf.scale).float().cuda()
    refined_rois = mutils.apply_box_deltas_2D(rois, deltas_specific * std_dev) * scale if cf.dim == 2 else \
        mutils.apply_box_deltas_3D(rois, deltas_specific * std_dev) * scale

    # round and cast to int since we're dealing with pixels now
    refined_rois = mutils.clip_to_window(cf.window, refined_rois)
    refined_rois = torch.round(refined_rois)

    # filter out low confidence boxes
    keep = idx
    keep_bool = (class_scores >= cf.model_min_confidence)
    if not 0 in torch.nonzero(keep_bool).size():

        score_keep = torch.nonzero(keep_bool)[:, 0]
        pre_nms_class_ids = class_ids[score_keep]
        pre_nms_rois = refined_rois[score_keep]
        pre_nms_scores = class_scores[score_keep]
        pre_nms_batch_ixs = batch_ixs[score_keep]

        for j, b in enumerate(mutils.unique1d(pre_nms_batch_ixs)):

            bixs = torch.nonzero(pre_nms_batch_ixs == b)[:, 0]
            bix_class_ids = pre_nms_class_ids[bixs]
            bix_rois = pre_nms_rois[bixs]
            bix_scores = pre_nms_scores[bixs]

            for i, class_id in enumerate(mutils.unique1d(bix_class_ids)):

                ixs = torch.nonzero(bix_class_ids == class_id)[:, 0]
                # nms expects boxes sorted by score.
                ix_rois = bix_rois[ixs]
                ix_scores = bix_scores[ixs]
                ix_scores, order = ix_scores.sort(descending=True)
                ix_rois = ix_rois[order, :]

                class_keep = nms.nms(ix_rois, ix_scores, cf.detection_nms_threshold)

                # map indices back.
                class_keep = keep[score_keep[bixs[ixs[order[class_keep]]]]]
                # merge indices over classes for current batch element
                b_keep = class_keep if i == 0 else mutils.unique1d(torch.cat((b_keep, class_keep)))

            # only keep top-k boxes of current batch-element
            top_ids = class_scores[b_keep].sort(descending=True)[1][:cf.model_max_instances_per_batch_element]
            b_keep = b_keep[top_ids]

            # merge indices over batch elements.
            batch_keep = b_keep  if j == 0 else mutils.unique1d(torch.cat((batch_keep, b_keep)))

        keep = batch_keep

    else:
        keep = torch.tensor([0]).long().cuda()

    # arrange output
    output = [refined_rois[keep], batch_ixs[keep].unsqueeze(1)]
    output += [class_ids[keep].unsqueeze(1).float(), class_scores[keep].unsqueeze(1)]

    result = torch.cat(output, dim=1)
    # shape: (n_keeps, catted feats), catted feats: [0:dim*2] are box_coords, [dim*2] are batch_ics,
    # [dim*2+1] are class_ids, [dim*2+2] are scores, [dim*2+3:] are regression vector features (incl uncertainty)
    # print('\n')
    # print(rois.shape[0],result.shape[0])
    return result

############################################################
#  Output Handler
############################################################

def refine_detections(anchors, probs, deltas, batch_ixs, cf):
    """
    Refine classified proposals, filter overlaps and return final
    detections. n_proposals here is typically a very large number: batch_size * n_anchors.
    This function is hence optimized on trimming down n_proposals.
    :param anchors: (n_anchors, 2 * dim)
    :param probs: (n_proposals, n_classes) softmax probabilities for all rois as predicted by classifier head.
    :param deltas: (n_proposals, n_classes, 2 * dim) box refinement deltas as predicted by bbox regressor head.
    :param batch_ixs: (n_proposals) batch element assignemnt info for re-allocation.
    :return: result: (n_final_detections, (y1, x1, y2, x2, (z1), (z2), batch_ix, pred_class_id, pred_score))
    """
    anchors = anchors.repeat(batch_ixs.unique().shape[0], 1)

    # flatten foreground probabilities, sort and trim down to highest confidences by pre_nms limit.
    fg_probs = probs[:, 1:].contiguous()
    flat_probs, flat_probs_order = fg_probs.view(-1).sort(descending=True)
    keep_ix = flat_probs_order[:cf.pre_nms_limit]
    # reshape indices to 2D index array with shape like fg_probs.
    keep_arr = torch.cat(((keep_ix / fg_probs.shape[1]).unsqueeze(1), (keep_ix % fg_probs.shape[1]).unsqueeze(1)), 1)

    pre_nms_scores = flat_probs[:cf.pre_nms_limit]
    pre_nms_class_ids = keep_arr[:, 1] + 1  # add background again.
    pre_nms_batch_ixs = batch_ixs[keep_arr[:, 0]]
    pre_nms_anchors = anchors[keep_arr[:, 0]]
    pre_nms_deltas = deltas[keep_arr[:, 0]]
    keep = torch.arange(pre_nms_scores.size()[0]).long().cuda()

    # apply bounding box deltas. re-scale to image coordinates.
    std_dev = torch.from_numpy(np.reshape(cf.rpn_bbox_std_dev, [1, cf.dim * 2])).float().cuda()
    scale = torch.from_numpy(cf.scale).float().cuda()
    refined_rois = mutils.apply_box_deltas_2D(pre_nms_anchors / scale, pre_nms_deltas * std_dev) * scale \
        if cf.dim == 2 else mutils.apply_box_deltas_3D(pre_nms_anchors / scale, pre_nms_deltas * std_dev) * scale

    # round and cast to int since we're deadling with pixels now
    refined_rois = mutils.clip_to_window(cf.window, refined_rois)
    pre_nms_rois = torch.round(refined_rois)
    for j, b in enumerate(mutils.unique1d(pre_nms_batch_ixs)):

        bixs = torch.nonzero(pre_nms_batch_ixs == b)[:, 0]
        bix_class_ids = pre_nms_class_ids[bixs]
        bix_rois = pre_nms_rois[bixs]
        bix_scores = pre_nms_scores[bixs]

        for i, class_id in enumerate(mutils.unique1d(bix_class_ids)):

            ixs = torch.nonzero(bix_class_ids == class_id)[:, 0]
            # nms expects boxes sorted by score.
            ix_rois = bix_rois[ixs]
            ix_scores = bix_scores[ixs]
            ix_scores, order = ix_scores.sort(descending=True)
            ix_rois = ix_rois[order, :]
            ix_scores = ix_scores

            class_keep = nms.nms(ix_rois, ix_scores, cf.detection_nms_threshold)

            # map indices back.
            class_keep = keep[bixs[ixs[order[class_keep]]]]
            # merge indices over classes for current batch element
            b_keep = class_keep if i == 0 else mutils.unique1d(torch.cat((b_keep, class_keep)))

        # only keep top-k boxes of current batch-element.
        top_ids = pre_nms_scores[b_keep].sort(descending=True)[1][:cf.model_max_instances_per_batch_element]
        b_keep = b_keep[top_ids]
        # merge indices over batch elements.
        batch_keep = b_keep if j == 0 else mutils.unique1d(torch.cat((batch_keep, b_keep)))

    keep = batch_keep

    # arrange output.
    result = torch.cat((pre_nms_rois[keep],
                        pre_nms_batch_ixs[keep].unsqueeze(1).float(),
                        pre_nms_class_ids[keep].unsqueeze(1).float(),
                        pre_nms_scores[keep].unsqueeze(1)), dim=1)

    return result



def get_results(cf, img_shape, detections, seg_logits, box_results_list=None):
    """
    Restores batch dimension of merged detections, unmolds detections, creates and fills results dict.
    :param img_shape:
    :param detections: (n_final_detections, (y1, x1, y2, x2, (z1), (z2), batch_ix, pred_class_id, pred_score)
    :param box_results_list: None or list of output boxes for monitoring/plotting.
    each element is a list of boxes per batch element.
    :return: results_dict: dictionary with keys:
             'boxes': list over batch elements. each batch element is a list of boxes. each box is a dictionary:
                      [[{box_0}, ... {box_n}], [{box_0}, ... {box_n}], ...]
             'seg_preds': pixel-wise class predictions (b, 1, y, x, (z)) with values [0, ..., n_classes] for
                          retina_unet and dummy array for retina_net.
    """
    detections = detections.cpu().data.numpy()
    batch_ixs = detections[:, cf.dim*2]
    detections = [detections[batch_ixs == ix] for ix in range(img_shape[0])]

    # for test_forward, where no previous list exists.
    if box_results_list is None:
        box_results_list = [[] for _ in range(img_shape[0])]

    for ix in range(img_shape[0]):

        if 0 not in detections[ix].shape:

            boxes = detections[ix][:, :2 * cf.dim].astype(np.int32)
            class_ids = detections[ix][:, 2 * cf.dim + 1].astype(np.int32)
            scores = detections[ix][:, 2 * cf.dim + 2]

            # Filter out detections with zero area. Often only happens in early
            # stages of training when the network weights are still a bit random.
            if cf.dim == 2:
                exclude_ix = np.where((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) <= 0)[0]
            else:
                exclude_ix = np.where(
                    (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 5] - boxes[:, 4]) <= 0)[0]

            if exclude_ix.shape[0] > 0:
                boxes = np.delete(boxes, exclude_ix, axis=0)
                class_ids = np.delete(class_ids, exclude_ix, axis=0)
                scores = np.delete(scores, exclude_ix, axis=0)

            if 0 not in boxes.shape:
                for ix2, score in enumerate(scores):
                    if score >= cf.model_min_confidence:
                        box_results_list[ix].append({'box_coords': boxes[ix2],
                                                     'box_score': score,
                                                     'box_type': 'det',
                                                     'box_pred_class_id': class_ids[ix2]})

    results_dict = {'boxes': box_results_list}
    if seg_logits is None:
        # output dummy segmentation for retina_net.
        results_dict['seg_preds'] = np.zeros(img_shape)[:, 0][:, np.newaxis]
    else:
        # output label maps for retina_unet.
        results_dict['seg_preds'] = F.softmax(seg_logits, 1).argmax(1).cpu().data.numpy()[:, np.newaxis].astype('uint8')

    return results_dict


############################################################
#  Retina (U-)Net Class
############################################################


class net(nn.Module):


    def __init__(self, cf, logger):

        super(net, self).__init__()
        self.cf = cf
        self.logger = logger
        self.build()
        if self.cf.weight_init is not None:
            logger.info("using pytorch weight init of type {}".format(self.cf.weight_init))
            mutils.initialize_weights(self)
        else:
            logger.info("using default pytorch weight init")

    def build(self):
        """
        Build Retina Net architecture.
        """

        # Image size must be dividable by 2 multiple times.
        h, w = self.cf.patch_size[:2]
        if h / 2 ** 5 != int(h / 2 ** 5) or w / 2 ** 5 != int(w / 2 ** 5):
            raise Exception("Image size must be dividable by 2 at least 5 times "
                            "to avoid fractions when downscaling and upscaling."
                            "For example, use 256, 320, 384, 448, 512, ... etc. ")

        # instanciate abstract multi dimensional conv class and backbone model.
        conv = mutils.NDConvGenerator(self.cf.dim)
        backbone = utils.import_module('bbone', self.cf.backbone_path)

        # build Anchors, FPN, Classifier / Bbox-Regressor -head
        self.np_anchors = mutils.generate_pyramid_anchors(self.logger, self.cf)
        self.anchors = torch.from_numpy(self.np_anchors).float().cuda()
        self.Fpn = backbone.FPN(self.cf, conv, operate_stride1=self.cf.operate_stride1)
        self.Classifier = Classifier(self.cf, conv)
        self.BBRegressor = BBRegressor(self.cf, conv)
        self.fp_reduce=Fp_reduce(self.cf,conv)
        self.final_conv = conv(self.cf.end_filts, self.cf.num_seg_classes, ks=1, pad=0, norm=None, relu=None)


    def train_forward(self, batch, **kwargs):
        """
        train method (also used for validation monitoring). wrapper around forward pass of network. prepares input data
        for processing, computes losses, and stores outputs in a dictionary.
        :param batch: dictionary containing 'data', 'seg', etc.
        :return: results_dict: dictionary with keys:
                'boxes': list over batch elements. each batch element is a list of boxes. each box is a dictionary:
                        [[{box_0}, ... {box_n}], [{box_0}, ... {box_n}], ...]
                'seg_preds': pixelwise segmentation output (b, c, y, x, (z)) with values [0, .., n_classes].
                'monitor_values': dict of values to be monitored.
        """
        img = batch['data']
        if "roi_labels" in batch.keys():
            raise Exception("Key for roi-wise class targets changed in v0.1.0 from 'roi_labels' to 'class_target'.\n"
                            "If you use DKFZ's batchgenerators, please make sure you run version >= 0.20.1.")
        gt_class_ids = batch['class_target']
        gt_boxes = batch['bb_target']
        gt_masks = batch['roi_masks']

        var_seg_ohe = torch.FloatTensor(mutils.get_one_hot_encoding(batch['seg'], self.cf.num_seg_classes)).cuda()
        var_seg = torch.LongTensor(batch['seg']).cuda()

        img = torch.from_numpy(img).float().cuda()
        batch_class_loss = torch.FloatTensor([0]).cuda()
        batch_bbox_loss = torch.FloatTensor([0]).cuda()

        # list of output boxes for monitoring/plotting. each element is a list of boxes per batch element.
        box_results_list = [[] for _ in range(img.shape[0])]
        # detections, class_logits, pred_deltas, seg_logits,fp_redcue_feature = self.forward(img)
        class_logits, pred_deltas, seg_logits, fp_redcue_feature = self.forward(img)

        # forword Fp_reduce
        self.fp_redcue_feature=[fp_redcue_feature]
        rpn_pred_probs=F.softmax(class_logits, 2)

        proposal_boxes, detections_fpd=self.forward_Fp_reduce(img,rpn_pred_probs,pred_deltas) #(img,rpn_pred_probs,rpn_pred_deltas)
        mrcnn_class_logits,  target_class_ids, mrcnn_target_deltas, target_mask, \
        sample_proposals = self.loss_samples_forward(gt_class_ids, gt_boxes, gt_masks)


        # loop over batch
        for b in range(img.shape[0]):

            # add gt boxes to results dict for monitoring.
            if len(gt_boxes[b]) > 0:
                for ix in range(len(gt_boxes[b])):
                    box_results_list[b].append({'box_coords': batch['bb_target'][b][ix],
                                                'box_label': batch['class_target'][b][ix], 'box_type': 'gt'})

                # match gt boxes with anchors to generate targets.
                anchor_class_match, anchor_target_deltas = mutils.gt_anchor_matching(
                    self.cf, self.np_anchors, gt_boxes[b], gt_class_ids[b])

                # add positive anchors used for loss to results_dict for monitoring.
                pos_anchors = mutils.clip_boxes_numpy(
                    self.np_anchors[np.argwhere(anchor_class_match > 0)][:, 0], img.shape[2:])
                for p in pos_anchors:
                    box_results_list[b].append({'box_coords': p, 'box_type': 'pos_anchor'})

            else:
                anchor_class_match = np.array([-1]*self.np_anchors.shape[0])
                anchor_target_deltas = np.array([0])

            # print(anchor_class_match.shape)
            anchor_class_match = torch.from_numpy(anchor_class_match).cuda()
            anchor_target_deltas = torch.from_numpy(anchor_target_deltas).float().cuda()

            # compute losses.
            class_loss, neg_anchor_ix = compute_class_loss(anchor_class_match, class_logits[b])
            bbox_loss = compute_bbox_loss(anchor_target_deltas, pred_deltas[b], anchor_class_match)

            # add negative anchors used for loss to results_dict for monitoring.
            neg_anchors = mutils.clip_boxes_numpy(
                self.np_anchors[np.argwhere(anchor_class_match.cpu().numpy() == -1)][neg_anchor_ix, 0],
                img.shape[2:])
            for n in neg_anchors:
                box_results_list[b].append({'box_coords': n, 'box_type': 'neg_anchor'})

            batch_class_loss += class_loss / img.shape[0]
            batch_bbox_loss += bbox_loss / img.shape[0]
        # add positive and negative roi samples used for mrcnn losses to output list for monitoring.
        if 0 not in sample_proposals.shape:
            rois = mutils.clip_to_window(self.cf.window, sample_proposals).cpu().data.numpy()
            for ix, r in enumerate(rois):
                box_results_list[int(r[-1])].append({'box_coords': r[:-1] * self.cf.scale,
                                                     'box_type': 'pos_class' if target_class_ids[
                                                                                    ix] > 0 else 'neg_class'})

        batch_class_loss = batch_class_loss
        batch_bbox_loss = batch_bbox_loss

        # compute mrcnn losses.
        fp_reduce_class_loss = compute_fp_reduce_class_loss(target_class_ids, mrcnn_class_logits)
        # mrcnn_bbox_loss = compute_mrcnn_bbox_loss(mrcnn_target_deltas, mrcnn_pred_deltas, target_class_ids)

        # mrcnn can be run without pixelwise annotations available (Faster R-CNN mode).
        # In this case, the mask_loss is taken out of training.
        # if not self.cf.frcnn_mode:
        #     mrcnn_mask_loss = compute_mrcnn_mask_loss(target_mask, mrcnn_pred_mask, target_class_ids)
        # else:
        #     mrcnn_mask_loss = torch.FloatTensor([0]).cuda()

        loss = batch_class_loss + batch_bbox_loss + fp_reduce_class_loss
        # loss = fp_reduce_class_loss

        # monitor RPN performance: detection count = the number of correctly matched proposals per fg-class.
        dcount = [list(target_class_ids.cpu().data.numpy()).count(c) for c in np.arange(self.cf.head_classes)[1:]]


        results_dict = get_results(self.cf, img.shape, detections_fpd, seg_logits, box_results_list)

        results_dict['torch_loss'] = loss
        results_dict['monitor_values'] = {'loss': loss.item(), 'class_loss': fp_reduce_class_loss.item()}

        # results_dict['logger_string'] = "loss: {0:.2f}, rpn_class: {1:.2f}, rpn_bbox: {2:.2f}, fp_reduce_class: {3:.2f},dcount {6}".format(loss.item(), batch_class_loss.item(),batch_bbox_loss.item(), fp_reduce_class_loss.item(),dcount)
        results_dict['logger_string'] = "loss: {0:.2f}, rpn_class: {1:.2f}, rpn_bbox: {2:.2f}, fp_reduce_class: {3:.2f}".format(loss.item(), batch_class_loss.item(),batch_bbox_loss.item(), fp_reduce_class_loss.item())
        return results_dict


    def test_forward(self, batch, **kwargs):
        """
        test method. wrapper around forward pass of network without usage of any ground truth information.
        prepares input data for processing and stores outputs in a dictionary.
        :param batch: dictionary containing 'data'
        :return: results_dict: dictionary with keys:
               'boxes': list over batch elements. each batch element is a list of boxes. each box is a dictionary:
                       [[{box_0}, ... {box_n}], [{box_0}, ... {box_n}], ...]
               'seg_preds': pixel-wise class predictions (b, 1, y, x, (z)) with values [0, ..., n_classes] for
                            retina_unet and dummy array for retina_net.
        """
        img = batch['data']
        img = torch.from_numpy(img).float().cuda()
        # detections, _, _, seg_logits = self.forward(img)
        class_logits, pred_deltas, seg_logits, fp_redcue_feature = self.forward(img)
        self.fp_redcue_feature = [fp_redcue_feature]
        rpn_pred_probs = F.softmax(class_logits.view(-1, class_logits.shape[-1]), 1)
        _, detections_fpd = self.forward_Fp_reduce(img, rpn_pred_probs, pred_deltas)

        results_dict = get_results(self.cf, img.shape, detections_fpd, seg_logits)
        return results_dict


    def forward(self, img):
        """
        forward pass of the model.
        :param img: input img (b, c, y, x, (z)).
        :return: rpn_pred_logits: (b, n_anchors, 2)
        :return: rpn_pred_deltas: (b, n_anchors, (y, x, (z), log(h), log(w), (log(d))))
        :return: batch_proposal_boxes: (b, n_proposals, (y1, x1, y2, x2, (z1), (z2), batch_ix)) only for monitoring/plotting.
        :return: detections: (n_final_detections, (y1, x1, y2, x2, (z1), (z2), batch_ix, pred_class_id, pred_score)
        :return: detection_masks: (n_final_detections, n_classes, y, x, (z)) raw molded masks as returned by mask-head.
        """
        # Feature extraction
        fpn_outs = self.Fpn(img)
        seg_logits = self.final_conv(fpn_outs[0])
        selected_fmaps = [fpn_outs[i + 2] for i in self.cf.pyramid_levels]

        # Loop through pyramid layers
        class_layer_outputs, bb_reg_layer_outputs = [], []  # list of lists
        for p in selected_fmaps:
            class_layer_outputs.append(self.Classifier(p))
            bb_reg_layer_outputs.append(self.BBRegressor(p))

        # Concatenate layer outputs
        # Convert from list of lists of level outputs to list of lists
        # of outputs across levels.
        # e.g. [[a1, b1, c1], [a2, b2, c2]] => [[a1, a2], [b1, b2], [c1, c2]]
        class_logits = list(zip(*class_layer_outputs))
        class_logits = [torch.cat(list(o), dim=1) for o in class_logits][0]
        bb_outputs = list(zip(*bb_reg_layer_outputs))
        bb_outputs = [torch.cat(list(o), dim=1) for o in bb_outputs][0]

        # # merge batch_dimension and store info in batch_ixs for re-allocation.
        # batch_ixs = torch.arange(class_logits.shape[0]).unsqueeze(1).repeat(1, class_logits.shape[1]).view(-1).cuda()
        # flat_class_softmax = F.softmax(class_logits.view(-1, class_logits.shape[-1]), 1)
        # flat_bb_outputs = bb_outputs.view(-1, bb_outputs.shape[-1])
        # detections = refine_detections(self.anchors, flat_class_softmax, flat_bb_outputs, batch_ixs, self.cf)

        # return detections, class_logits, bb_outputs, seg_logits, fpn_outs[0]
        return class_logits, bb_outputs, seg_logits, fpn_outs[2]


    def forward_Fp_reduce(self, img,rpn_pred_probs,rpn_pred_deltas,is_training=True):
        """
               :param img: input images (b, c, y, x, (z)).
               :return: rpn_pred_logits: (b, n_anchors, 2)
               :return: rpn_pred_deltas: (b, n_anchors, (y, x, (z), log(h), log(w), (log(d))))
               :return: batch_proposal_boxes: (b, n_proposals, (y1, x1, y2, x2, (z1), (z2), batch_ix)) only for monitoring/plotting.
               :return: detections: (n_final_detections, (y1, x1, y2, x2, (z1), (z2), batch_ix, pred_class_id, pred_score)
               :return: detection_masks: (n_final_detections, n_classes, y, x, (z)) raw molded masks as returned by mask-head.
        """
        # generate proposals: apply predicted deltas to anchors and filter by foreground scores from RPN classifier.
        proposal_count = self.cf.post_nms_rois_training if is_training else self.cf.post_nms_rois_inference
        batch_rpn_rois, batch_proposal_boxes = refine_proposals(rpn_pred_probs, rpn_pred_deltas, proposal_count,self.anchors, self.cf)

        # merge batch dimension of proposals while storing allocation info in coordinate dimension.
        batch_ixs = torch.from_numpy(
            np.repeat(np.arange(batch_rpn_rois.shape[0]), batch_rpn_rois.shape[1])).float().cuda()
        rpn_rois = batch_rpn_rois.view(-1, batch_rpn_rois.shape[2])
        self.rpn_rois_batch_info = torch.cat((rpn_rois, batch_ixs.unsqueeze(1)), dim=1)

        # this is the first of two forward passes in the second stage, where no activations are stored for backprop.
        # here, all proposals are forwarded (with virtual_batch_size = batch_size * post_nms_rois.)
        # for inference/monitoring as well as sampling of rois for the loss functions.
        # processed in chunks of roi_chunk_size to re-adjust to gpu-memory.
        chunked_rpn_rois = self.rpn_rois_batch_info.split(self.cf.roi_chunk_size)
        class_logits_list, bboxes_list = [], []
        with torch.no_grad():
            for chunk in chunked_rpn_rois:
                # chunk_class_logits, chunk_bboxes = self.fp_reduce(self.fp_redcue_feature, chunk)
                # class_logits_list.append(chunk_class_logits)
                # bboxes_list.append(chunk_bboxes)
                chunk_class_logits = self.fp_reduce(self.fp_redcue_feature, chunk)
                class_logits_list.append(chunk_class_logits)
        batch_mrcnn_class_logits = torch.cat(class_logits_list, 0)
        self.batch_mrcnn_class_scores = F.softmax(batch_mrcnn_class_logits, dim=1) # (n_proposals, n_classes)

        # refine classified proposals, filter and return final detections.
        # detections = fp_refine_detections(self.cf, batch_ixs, rpn_rois, batch_mrcnn_bbox, self.batch_mrcnn_class_scores)
        batch_mrcnn_bbox=torch.zeros((self.batch_mrcnn_class_scores.size()[0],self.batch_mrcnn_class_scores.size()[1],2*self.cf.dim)).cuda()   # (n_proposals, n_classes, 2 * dim)
        detections = fp_refine_detections(self.cf, batch_ixs, rpn_rois, batch_mrcnn_bbox, self.batch_mrcnn_class_scores)
        # detections = fp_get_detections(self.cf, batch_ixs, rpn_rois, batch_mrcnn_bbox, self.batch_mrcnn_class_scores)

        # forward remaining detections through mask-head to generate corresponding masks.
        # scale = [img.shape[2]] * 4 + [img.shape[-1]] * 2
        # scale = torch.from_numpy(np.array(scale[:self.cf.dim * 2] + [1])[None]).float().cuda()

        # detection_boxes = detections[:, :self.cf.dim * 2 + 1] / scale
        # TODO: 这里加入mask或许可以，因为可以训练到的样本更多，而不是处理后的结果
        # with torch.no_grad():
        #     detection_masks = self.mask(self.fp_redcue_feature, detection_boxes)

        # return [batch_proposal_boxes, detections, detection_masks]
        return [batch_proposal_boxes, detections]

    def loss_samples_forward(self, batch_gt_class_ids, batch_gt_boxes, batch_gt_masks):
        """
        this is the second forward pass through the second stage (features from stage one are re-used).
        samples few rois in detection_target_layer and forwards only those for loss computation.
        :param batch_gt_class_ids: list over batch elements. Each element is a list over the corresponding roi target labels.
        :param batch_gt_boxes: list over batch elements. Each element is a list over the corresponding roi target coordinates.
        :param batch_gt_masks: list over batch elements. Each element is binary mask of shape (n_gt_rois, y, x, (z), c)
        :return: sample_logits: (n_sampled_rois, n_classes) predicted class scores.
        :return: sample_boxes: (n_sampled_rois, n_classes, 2 * dim) predicted corrections to be applied to proposals for refinement.
        :return: sample_mask: (n_sampled_rois, n_classes, y, x, (z)) predicted masks per class and proposal.
        :return: sample_target_class_ids: (n_sampled_rois) target class labels of sampled proposals.
        :return: sample_target_deltas: (n_sampled_rois, 2 * dim) target deltas of sampled proposals for box refinement.
        :return: sample_target_masks: (n_sampled_rois, y, x, (z)) target masks of sampled proposals.
        :return: sample_proposals: (n_sampled_rois, 2 * dim) RPN output for sampled proposals. only for monitoring/plotting.
        """
        # sample rois for loss and get corresponding targets for all Mask R-CNN head network losses.
        sample_ix, sample_target_class_ids, sample_target_deltas, sample_target_mask = \
            detection_target_layer(self.rpn_rois_batch_info, self.batch_mrcnn_class_scores,
                                   batch_gt_class_ids, batch_gt_boxes, batch_gt_masks, self.cf)

        # re-use feature maps and RPN output from first forward pass.
        sample_proposals = self.rpn_rois_batch_info[sample_ix]
        if 0 not in sample_proposals.size():
            sample_logits = self.fp_reduce(self.fp_redcue_feature, sample_proposals)
            # sample_mask = self.mask(self.fp_redcue_feature, sample_proposals)
        else:
            sample_logits = torch.FloatTensor().cuda()
            # sample_boxes = torch.FloatTensor().cuda()
            # sample_mask = torch.FloatTensor().cuda()

        # return [sample_logits, sample_boxes, sample_mask, sample_target_class_ids, sample_target_deltas,
        #         sample_target_mask, sample_proposals]

        return [sample_logits, sample_target_class_ids, sample_target_deltas,
                sample_target_mask, sample_proposals]

