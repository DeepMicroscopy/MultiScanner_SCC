from fastai.vision.all import *
from matplotlib import cm
from matplotlib.colors import ListedColormap
from einops import rearrange, reduce, repeat
from collections import defaultdict


COLORS = np.array([[128, 128, 128], # Excluded
          [255, 255, 255],  # BG
          [0, 0, 255],  # Normal
          [255, 128, 0]])  # Tumor

def get_item(slide_container):
    xmin, ymin = slide_container.get_new_train_coordinates()
    patch = slide_container.get_patch(xmin, ymin)
    y_patch = slide_container.get_y_patch(xmin, ymin) + 1
    return (patch, y_patch)

@typedispatch
def show_batch(x, y, samples, ctxs=None, max_n=6, nrows=None, ncols=1, figsize=None, **kwargs):
    if figsize is None: figsize = (ncols*12, min(x[0].shape[0], max_n) * 3)
    if ctxs is None: ctxs = get_grid(min(x[0].shape[0], max_n), nrows=min(x[0].shape[0], max_n), ncols=ncols, figsize=figsize)
    for i,ctx in enumerate(ctxs):
            image= tensor(x[i])
            line = image.new_zeros(image.shape[0], image.shape[1], 5)
            mask = y[i]
            overlay = tensor(np.asarray([COLORS[c] for c in np.int16(mask.flatten())],dtype=np.uint8).reshape((mask.shape[0], mask.shape[1], -1))).permute(2,0,1)
            show_image(torch.cat([image, line, overlay], dim=2), ctx=ctx, **kwargs)

class customDataLoader(TfmdDL):
    def show_results(self,
        b, # Batch to show results for
        out, # Predicted output from model for the batch
        max_n:int=9, # Maximum number of items to show
        ctxs=None, # List of `ctx` objects to show data. Could be matplotlib axis, DataFrame etc
        show:bool=True, # Whether to display data
        **kwargs
    ):
        x, y, its = self.show_batch(b, max_n=max_n, show=False)
        seg_out = torch.max(out, dim=1)[1]+ 1
        b_out = type(b)(b[:self.n_inp] + (seg_out,))
        _types = self._types[first(self._types.keys())]
        b_out = tuple([cast(x, typ) for x,typ in zip(b_out, _types)])
        b_out = to_cpu(self.after_batch.decode(b_out))
        if not show:
            return (x,y,its,b_out[self.n_inp:])
        show_results(b_out[:self.n_inp], b_out[self.n_inp:], ctxs=ctxs, max_n=max_n, **kwargs)


def generate_dataloaders(train_files, valid_files, patches_per_slide, batch_size, mean, std):
    dblock = DataBlock(blocks=(ImageBlock, MaskBlock),
                       dl_type=customDataLoader,
                       splitter=TrainTestSplitter(test_size=len(valid_files) * patches_per_slide, shuffle=False),
                       get_items=lambda files: [get_item(file) for file in files for _ in range(patches_per_slide)],
                       getters=[ItemGetter(0), ItemGetter(1)], item_tfms=[], batch_tfms=[
            Normalize.from_stats(torch.FloatTensor(mean),
                                 torch.FloatTensor(std)),
                                 *aug_transforms()], n_inp=1)
    dls = dblock.dataloaders(train_files + valid_files, bs=batch_size, num_workers=0)
    return dls