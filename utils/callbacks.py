import torch
from fastai.metrics import Metric, AvgMetric
from fastai.callback.core import Callback
from slide.slide_helper import generate_dataloaders
from torchmetrics import JaccardIndex


class IoU(AvgMetric):
    def __init__(self):
        super().__init__(func=JaccardIndex(num_classes=4, ignore_index=0))
        self._name = "iou"
    def accumulate(self, learn):
        bs = learn.yb[0].shape[0]
        if bs > 0:
            self.total += self.func.to(learn.yb[0].device)(torch.max(learn.pred, dim=1)[1] + 1, learn.yb[0])*bs
            self.count += bs
    @property
    def name(self):  return self._name


class LossComponent(Metric):
    def __init__(self, name):
        self._name = name
    def reset(self):           self.total,self.count = 0.,0
    def accumulate(self, learn):
        bs = learn.yb[0].shape[0]
        self.total += learn.loss_func.metrics[self._name]*bs
        self.count += bs
    @property
    def value(self): return self.total/self.count if self.count != 0 else None
    @property
    def name(self):  return self._name

def LossComponents(components):
    return [LossComponent(loss) for loss in components]
    
class ResetDataloaders(Callback):
    def __init__(self, train_files, valid_files, patches_per_slide, batch_size, mean, std):
        super().__init__()
        self.train_files = train_files
        self.valid_files = valid_files
        self.patches_per_slide = patches_per_slide
        self.batch_size = batch_size
        self.mean = mean
        self.std = std

    def after_epoch(self):
        dls = generate_dataloaders(self.train_files, self.valid_files, self.patches_per_slide,self.batch_size,self.mean,self.std)
        self.learn.dls.train = dls.train