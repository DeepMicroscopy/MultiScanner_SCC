from slide.process_slides import *
import hydra
from omegaconf import DictConfig
from slide.slide_helper import *
from cpbd import compute
import cv2
import seaborn as sns


def kde_plots(files, scanner):
    reds = np.concatenate([t.thumbnail[:, :, 0][np.logical_not(t.mask)].flatten() for t in files])
    greens = np.concatenate([t.thumbnail[:, :, 1][np.logical_not(t.mask)].flatten() for t in files])
    blues = np.concatenate([t.thumbnail[:, :, 2][np.logical_not(t.mask)].flatten() for t in files])
    data = pd.DataFrame(np.vstack((reds, greens, blues)).T)
    plt.figure(figsize=(10, 10))
    kde = sns.kdeplot(data, palette=["tomato", "yellowgreen", "cornflowerblue"], legend=False, multiple="stack")
    kde.set(xlim=(0, 255))
    kde.set(ylim=(0, 0.04))
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.ylabel('Density', fontsize=25)
    plt.savefig("{}_hist.pdf".format(scanner), bbox_inches='tight')


@hydra.main(version_base=None,config_path="configs/",  config_name='gt450')
def qc(cfg: DictConfig):
    # Get training parameters from config file
    patch_size = cfg.data.patch_size
    level = cfg.data.level
    label_dict = {'Bg': 0, 'Bone': 1, 'Cartilage': 1, 'Dermis': 1, 'Epidermis': 1, 'Subcutis': 1, 'Inflamm/Necrosis': 1,
                'Melanoma': 2, 'Plasmacytoma': 2,'Mast Cell Tumor': 2, 'PNST': 2, 'SCC': 2, 'Trichoblastoma': 2, 'Histiocytoma': 2}

    # Load images
    train_files, valid_files, test_files = load_slides(patch_size,label_dict, level, image_path = Path(cfg.files.image_path), annotation_file = Path(cfg.files.annotation_file), scanner=cfg.training.scanner)

    # Collect statistics per slide
    red, green, blue, sharpness, contrast = [], [], [], [], []
    for file in train_files+valid_files+test_files:
        red.append(np.mean(file.thumbnail[:, :, 0][np.logical_not(file.mask)].flatten()))
        green.append(np.mean(file.thumbnail[:, :, 1][np.logical_not(file.mask)].flatten()))
        blue.append(np.mean(file.thumbnail[:, :, 2][np.logical_not(file.mask)].flatten()))
        grayscale = cv2.cvtColor(file.thumbnail, cv2.COLOR_RGB2GRAY)
        grayscale[grayscale == 0] = 255
        # Calculate sharpness as cumulative probability of blur detection
        sharpness.append(compute(grayscale))
        # Compute Michelson Contrast
        contrast.append((float(grayscale[np.logical_not(file.mask)].max()) - float(
            grayscale[np.logical_not(file.mask)].min())) / (float(grayscale[np.logical_not(file.mask)].max()) + float(
            grayscale[np.logical_not(file.mask)].min())))

    # Compute dataset statistics
    print("Red:", np.round(np.mean(red), 2), "+-", np.round(np.std(red), 2))
    print("Green:", np.round(np.mean(green), 2), "+-", np.round(np.std(green), 2))
    print("Blue:", np.round(np.mean(blue), 2), "+-", np.round(np.std(blue), 2))
    print("Sharpness:", np.round(np.mean(sharpness), 2), "+-", np.round(np.std(sharpness), 2))
    print("Contrast:", np.round(np.mean(contrast), 2), "+-", np.round(np.std(contrast), 2))

    #kde_plots(train_files+valid_files+test_files, cfg.training.scanner)

if __name__ == '__main__':
    qc()




