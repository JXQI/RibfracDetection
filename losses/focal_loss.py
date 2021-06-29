import torch.nn.functional as F
import torch
import numpy as np
import utils.model_utils as mutils

def sigmoid_focal_loss(predict,target,gamma=2,alpha=None,reduction="sum",shem_poolsize=20):
    """
    predict_logits:logits from classifier sub-network.
    target:n anchor [-1,0,class_num] for negative, neutral, and positive matched anchors.
    """
    # Positive and Negative anchors contribute to the loss,
    # but neutral anchors (match value = 0) don't.
    target=target.long()
    pos_indices = torch.nonzero(target > 0)
    neg_indices = torch.nonzero(target == -1)

    # neg_indices=neg_indices[:100]
    # if 0 not in pos_indices.shape:
    #     pos_indices = pos_indices.squeeze(1)
    #     roi_logits_pos = predict[pos_indices]
    #     targets_pos = target[pos_indices]

    if 0 not in neg_indices.shape:
        neg_indices = neg_indices.squeeze(1)
        roi_logits_neg = predict[neg_indices]
        # targets_neg = target[neg_indices]
        # targets_neg[targets_neg == -1] = 0
        negative_count = 100*np.max((1, pos_indices.shape[0]))
        roi_probs_neg = F.softmax(roi_logits_neg, dim=1)
        neg_ix = mutils.shem(roi_probs_neg, negative_count, shem_poolsize)

        roi_logits_neg=roi_logits_neg[neg_ix]
        targets_neg=torch.LongTensor([0] * neg_ix.shape[0]).cuda()

    if 0 not in pos_indices.shape:
        pos_indices = pos_indices.squeeze(1)
        roi_logits_pos = predict[pos_indices]
        targets_pos = target[pos_indices]
        # print(len(targets_pos),len(targets_neg))
        targets=torch.cat((targets_pos,targets_neg),dim=0)
        predict_logit=torch.cat((roi_logits_pos,roi_logits_neg),dim=0)
    else:
        targets=targets_neg
        predict_logit=roi_logits_neg

    # one hot
    targets=targets.reshape((-1,1))
    uu=torch.zeros(targets.shape[0],2).cuda()
    targets=uu.scatter_(1,targets,1)
    predict_logit=F.softmax(predict_logit,dim=1)
    predict_logit=predict_logit.clamp(min=0.0001,max=1.0)
    # print(targets.shape,predict_logit.shape)
    alpha=torch.tensor([0.75,0.25]).cuda()
    if alpha is not None:
        loss=-alpha*torch.pow(1-predict_logit,gamma)*predict_logit.log()*targets
    else:
        loss=-torch.pow(1-predict_logit,gamma)*predict_logit.log()*targets
    # print(loss)
    # ce_loss=F.binary_cross_entropy_with_logits(predict_logit,targets,reduction='none')
    # # ce_loss = F.cross_entropy(predict_logit, targets, reduction='none')
    # predict_logit=torch.sigmoid(predict_logit)
    # p_t=targets*predict_logit+(1-predict_logit)*(1-targets)
    # loss=ce_loss*((1-p_t)**gamma)
    #
    # if alpha >= 0:
    #     alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    #     loss = alpha_t * loss
    #
    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    # 保证程序不报错
    neg_anchor_ix = np.array([]).astype('int32')

    return loss,neg_anchor_ix



