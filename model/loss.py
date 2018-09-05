import torch
import torch.nn as nn


class LossBinary:
    """
     Implementation from  https://github.com/ternaus/robot-surgery-segmentation
     loss = BCE - weight * IoU
    """

    def __init__(self, jaccard_weight=0):
        self.nll_loss = nn.BCEWithLogitsLoss()
        self.jaccard_weight = jaccard_weight

    def __call__(self, outputs, targets):
        loss = self.nll_loss(outputs, targets)

        if self.jaccard_weight:
            eps = 1e-15
            jaccard_target = (targets == 1.0).float()
            jaccard_output = torch.sigmoid(outputs)

            intersection = (jaccard_output * jaccard_target).sum()
            union = jaccard_output.sum() + jaccard_target.sum()

            loss -= self.jaccard_weight * torch.log((intersection + eps) / (union - intersection + eps))
        return loss


def get_jaccard(y_true, y_pred):
    """calculate IoU

    Arguments:
        y_true {tensor (b, c, h, w)} -- true mask
        y_pred {tensor (b, c, h, w)} -- predictate mask

    Returns:
        [tensor scaler] -- [batch mean IoU]
    """
    epsilon = 1e-15
    intersection = (y_pred * y_true).sum(dim=-2).sum(dim=-1).sum(dim=-1)
    union = y_true.sum(dim=-2).sum(dim=-1).sum(dim=-1) + y_pred.sum(dim=-2).sum(dim=-1).sum(dim=-1)

    return (intersection / (union - intersection + epsilon)).mean()
