import torch.nn.functional as F
import torch

def sigmoid_focal_loss(predict,target,gamma=2,alpha=0.25,reduction="mean"):
    """
    predict_logits:logits from classifier sub-network.
    target:n anchor [-1,0,class_num] for negative, neutral, and positive matched anchors.
    """
    # Positive and Negative anchors contribute to the loss,
    # but neutral anchors (match value = 0) don't.
    pos_indices = torch.nonzero(target > 0)
    neg_indices = torch.nonzero(target == -1)

    if 0 not in pos_indices.shape:
        pos_indices = pos_indices.squeeze(1)
        roi_logits_pos = predict[pos_indices]
        targets_pos = target[pos_indices]

    if 0 not in neg_indices.shape:
        neg_indices = neg_indices.squeeze(1)
        roi_logits_neg = predict[neg_indices]
        targets_neg=target[neg_indices]
        targets_neg[targets_neg==-1]=0

    targets=torch.cat((targets_pos,targets_neg),dim=0)
    predict_logit=torch.cat((roi_logits_pos,roi_logits_neg),dim=0)

    ce_loss=F.binary_cross_entropy_with_logits(predict_logit,targets,reduction='none')
    p_t=targets*predict_logit+(1-predict_logit)*(1-targets)
    loss=ce_loss*((1-p_t)**gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss,neg_indices[:len(pos_indices)]



