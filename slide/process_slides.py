import numpy as np
import pandas as pd
import glob
from pathlib import Path
from slide.slide_container import SlideContainer
from tqdm import tqdm


def create_indices(files, patches_per_slide):
    indices = []
    for i, file in enumerate(files):
        indices += patches_per_slide * [i]
    return indices

def load_slides(patch_size=256, label_dict=None, level = None, image_path =None, annotation_file=None, scanner="aperio"):
    train_files = []
    valid_files = []
    test_files = []

    slides = pd.read_csv('scc_dataset.csv', delimiter=";")
    for index, row in tqdm(slides.iterrows()):
        image_file = Path(glob.glob("{}/{}_{}.tif".format(str(image_path), row["Slide"], scanner), recursive=True)[0])
        if row["Dataset"] == "train":
            train_files.append(SlideContainer(image_file, annotation_file, level, patch_size, patch_size, label_dict = label_dict))
        elif row["Dataset"] == "val":
            valid_files.append(SlideContainer(image_file, annotation_file, level, patch_size, patch_size, label_dict = label_dict))
        elif row["Dataset"] == "test":
            test_files.append(SlideContainer(image_file, annotation_file, level, patch_size, patch_size, label_dict = label_dict))
        else:
            pass

    return train_files, valid_files, test_files