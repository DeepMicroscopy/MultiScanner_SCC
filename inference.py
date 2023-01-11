from slide.slide_helper import *
from slide.process_slides import *
import hydra
from omegaconf import DictConfig
from torchvision import models
from torchvision import transforms
from einops import rearrange
from torchmetrics import ConfusionMatrix
import os
from fastai.vision import *


def stitch_output_mask(filename, scanner):
    tile_size = (64, 64)
    max_x, max_y = 0, 0
    tile_paths = []
    for file in glob.glob(os.getcwd() + "\\temp\\*"):
        path = Path(file)
        parts = path.stem.split("_")
        if path.suffix == '.png':
            x, y = parts[-2:]
            tile_paths.append((path, int(x), int(y)))
            max_x = max(max_x, int(x))
            max_y = max(max_y, int(y))

    new_size = (max_x // 4, max_y // 4)
    # Create an output image
    output = Image.new('RGB', new_size)

    for path, x, y in tile_paths:
        tile = Image.open(path)
        tile = tile.resize(tile_size)
        output.paste(tile, (x // 4, y // 4))

    output.save("{}_{}.jpg".format(filename, scanner))


def segmentation_inference_center_crop(files, model, device, scanner, n_classes,  stats=None):
    cm = np.zeros((n_classes, n_classes))
    CM = ConfusionMatrix(num_classes=n_classes + 1)
    for slide_container in files:
        os.mkdir(Path(os.getcwd() + "//temp"))
        shape = slide_container.slide.level_dimensions[slide_container.level]
        x_indices = np.arange(0, int((shape[0] // (slide_container.width//2)) + 1)) * (slide_container.width//2)
        y_indices = np.arange(0, int((shape[1] // (slide_container.height//2)) + 1)) * (slide_container.height//2)

        with torch.no_grad():
            # segmentation inference
            model.eval()
            for y in tqdm(y_indices,desc='Processing %s' % slide_container.file.stem):
                x_loader = DataLoader(x_indices, batch_size=8)
                for xs in x_loader:
                    images = [slide_container.get_patch(x,y) for x in xs]
                    gts = torch.stack([image2tensor(slide_container.get_y_patch(x,y)) for x in xs])
                    if stats:
                        tensors = [transforms.Normalize(*stats)(image2tensor(img / 255.)) for img in images]
                    else:
                        tensors = [image2tensor(img / 255.) for img in images]
                    input_batch = torch.stack(tensors)
                    preds = model(input_batch.to(device))
                    outputs = torch.max(torch.softmax(preds, dim=1),dim=1)[1]
                    start, stop = int(slide_container.width*(1/4)), int(slide_container.width*(3/4))
                    update = CM(outputs[:, start:stop, start:stop].flatten().cpu().long() + 1, gts[:, :, start:stop, start:stop].flatten().long() + 1)
                    cm += update[1:,1:].numpy()
                    for o in range(outputs.shape[0]):
                        plt.imsave("temp/{}_{}_{}.png".format(slide_container.file.stem, xs[o], y),outputs[o][start:stop, start:stop].cpu().numpy(), vmin=0, vmax=2)
        stitch_output_mask(slide_container.file.stem, scanner)
        shutil.rmtree(Path(os.getcwd() + "//temp"))
    return cm



@hydra.main(version_base=None,config_path="configs/", config_name='cs2')
def main(cfg: DictConfig):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    patch_size = cfg.data.patch_size
    level = cfg.data.level
    mean = np.array(cfg.data.mean.split(","), dtype=np.float32)
    std = np.array(cfg.data.std.split(","), dtype=np.float32)
    cms = np.zeros((5, 3, 3))

    label_dict = {'Bg': 0, 'Bone': 1, 'Cartilage': 1, 'Dermis': 1, 'Epidermis': 1, 'Subcutis': 1, 'Inflamm/Necrosis': 1,
                  'Melanoma': 2, 'Plasmacytoma': 2,'Mast Cell Tumor': 2, 'PNST': 2, 'SCC': 2, 'Trichoblastoma': 2, 'Histiocytoma': 2}
    model = create_unet_model(models.resnet18, n_out=3, img_size=(patch_size, patch_size))
    model.load_state_dict(torch.load('models/{}.pth'.format(cfg.training.scanner), map_location=device))
    stats = (torch.FloatTensor(mean), torch.FloatTensor(std))

    for s, scanner in enumerate(["cs2", "nz20", "nz210", "3dhistech", "gt450"]):
        _, _, test_files = load_slides(patch_size,label_dict, level, image_path = Path(cfg.files.image_path), annotation_file = Path(cfg.files.annotation_file), scanner=scanner)
        cms[s] = segmentation_inference_center_crop(test_files, model.to(device), device, scanner=cfg.training.scanner,n_classes=max(label_dict.values()) + 1, stats=stats)

    with pd.ExcelWriter('{}.xlsx'.format(cfg.training.scanner), engine='xlsxwriter') as writer:
        for cm, scanner in zip(cms, ["cs2", "nz20", "nz210", "3dhistech", "gt450"]):
            df = pd.DataFrame(cm, index=["BG", "Normal", "Tumor"], columns=["BG", "Normal", "Tumor"])
            df.to_excel(writer, sheet_name=scanner)

if __name__ == '__main__':
    main()
