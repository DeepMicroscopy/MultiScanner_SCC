from slide.process_slides import *
import hydra
from omegaconf import DictConfig
from torchvision import models
from utils.combo_loss import ComboLoss
from utils.callbacks import ResetDataloaders, IoU, LossComponents
from slide.slide_helper import *


def random_seed(seed_value, use_cuda):
  '''
  Sets the random seed for numpy, pytorch, python.random and pytorch GPU vars.
  '''
  np.random.seed(seed_value) # Numpy vars
  torch.manual_seed(seed_value) # PyTorch vars
  random.seed(seed_value) # Python
  if use_cuda: # GPU vars
      torch.cuda.manual_seed(seed_value)
      torch.cuda.manual_seed_all(seed_value)
      torch.backends.cudnn.deterministic = True
      torch.backends.cudnn.benchmark = False
  print(f'Random state set:{seed_value}, cuda used: {use_cuda}')


@hydra.main(version_base=None,config_path="configs/",  config_name='gt450')
def train(cfg: DictConfig):
    # Confirm that you have a GPU!
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Get training parameters from config file
    patch_size = cfg.data.patch_size
    batch_size = cfg.data.batch_size
    level = cfg.data.level
    mean = np.array(cfg.data.mean.split(","), dtype=np.float32)
    std = np.array(cfg.data.std.split(","), dtype=np.float32)
    label_dict = {'Bg': 0, 'Bone': 1, 'Cartilage': 1, 'Dermis': 1, 'Epidermis': 1, 'Subcutis': 1, 'Inflamm/Necrosis': 1,
                'Melanoma': 2, 'Plasmacytoma': 2,'Mast Cell Tumor': 2, 'PNST': 2, 'SCC': 2, 'Trichoblastoma': 2, 'Histiocytoma': 2}

    # Initialize optimizer and seed for reproducibility
    opt_func = partial(Adam, mom=0.9, sqr_mom=0.99, eps=1e-05, wd=0.01, decouple_wd=True)
    random_seed(cfg.training.seed, torch.cuda.is_available())

    # Load images and generate dataloaders
    train_files, valid_files, test_files = load_slides(patch_size,label_dict, level, image_path = Path(cfg.files.image_path), annotation_file = Path(cfg.files.annotation_file), scanner=cfg.training.scanner)
    dls = generate_dataloaders(train_files, valid_files, cfg.data.patches_per_slide, batch_size, mean=mean, std=std)
    #dls.train.show_batch()

    # Initialize learner + loss
    loss_func = ComboLoss()
    learn = unet_learner(dls, models.resnet18, n_out=max(label_dict.values()) + 1, opt_func=opt_func, loss_func=loss_func,
                   metrics=[*LossComponents(loss_func.metrics.keys()),IoU()],
    cbs=[ResetDataloaders(train_files, valid_files, cfg.data.patches_per_slide, batch_size, mean, std),
    SaveModelCallback(monitor='iou', fname="bestmodel_{}".format(cfg.training.scanner))])

    # Train
    learn.fit_one_cycle(2, cfg.training.lr)

if __name__ == '__main__':
    train()




