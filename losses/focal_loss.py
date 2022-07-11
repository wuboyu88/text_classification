import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    # todo 当样本不平衡的时候用focal loss https://zhuanlan.zhihu.com/p/49981234
    # cross_entropy_loss = log_softmax+nll_loss
    """Multi-class Focal loss implementation"""

    def __init__(self, gamma=2, weight=None, ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, input_, target):
        """
        input_: [N, C]
        target: [N, ]
        """
        logpt = F.log_softmax(input_, dim=1)
        pt = torch.exp(logpt)
        logpt = (1 - pt) ** self.gamma * logpt
        loss = F.nll_loss(logpt, target, self.weight, ignore_index=self.ignore_index)
        return loss
