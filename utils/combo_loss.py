import torch
from torch import nn
import torch.nn.functional as F


class CELoss(nn.modules.loss._WeightedLoss):
    def __init__(self, weight=None, gamma=2,reduction='mean', ignore_index=-1):
        super(CELoss, self).__init__(weight,reduction=reduction)
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.weight = weight
        self.name = 'ce_loss'

    def forward(self, output, target):
        ce_loss = F.cross_entropy(output, target, reduction='none', weight=self.weight,ignore_index=self.ignore_index)
        if self.reduction == 'sum':
            ce_loss = ce_loss.sum()
        else:
            ce_loss = ce_loss.mean()
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss)
        return ce_loss #focal_loss


class DiceLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, weight=None, reduction='mean', ignore_index=-1):
        super(DiceLoss, self).__init__(weight,reduction=reduction)
        self.ignore_index = ignore_index
        self.weight = weight
        self.name = 'dice_loss'

    def forward(self, output, target):
        eps = 0.0001
        output = torch.softmax(output, dim=1)
        encoded_target = output.detach() * 0
        if self.ignore_index is not None:
            mask = target == self.ignore_index
            target = target.clone()
            target[mask] = 0
            encoded_target.scatter_(1, target.unsqueeze(1), 1)
            mask = mask.unsqueeze(1).expand_as(encoded_target)
            encoded_target[mask] = 0
        else:
            encoded_target.scatter_(1, target.unsqueeze(1), 1)
        weights = self.weight if self.weight is not None else 1.0

        intersection = output * encoded_target
        numerator = intersection.sum(0)
        denominator = output + encoded_target

        if self.ignore_index is not None:
            denominator[mask] = 0
        denominator = denominator.sum(0)
        return 1-((2*(weights*numerator).sum() + eps)/(weights*denominator).sum() + eps)


class ComboLoss(nn.Module):
    def __init__(self, reduction='mean', loss_funcs=[CELoss(), DiceLoss()], loss_wts=[1, 1], ch_wts=[1, 1, 1]):
        super().__init__()
        self.reduction = reduction
        self.ch_wts = ch_wts
        self.loss_wts = loss_wts
        self.loss_funcs = loss_funcs
        self.metrics = dict.fromkeys([loss.name for loss in self.loss_funcs], 0)

    def forward(self, output, labels):
        output = output.transpose(1, -1).contiguous()
        labels = labels.transpose(1,-1).contiguous().view(-1)
        output = output.view(-1,output.shape[-1])

        for loss_func in self.loss_funcs:
            loss_func.reduction = self.reduction
        loss = 0
        assert len(self.loss_wts) == len(self.loss_funcs)
        for loss_wt, loss_func in zip(self.loss_wts, self.loss_funcs):
            l = loss_wt*loss_func(output, labels-1)
            loss += l
            self.metrics.update({loss_func.name:l})
        return loss
