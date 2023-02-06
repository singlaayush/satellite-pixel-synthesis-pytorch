import cv2
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np
import fire

def to_remove(path):
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return image.mean() == 63.0  # all pixels in erroneous imagery are 63, can use that fact to filter
#to_remove("/deep/group/aicc-bootcamp/transportation/data/dvrpc/aerial/imagery/500:0:0.1:18:-769:88110.png") # to verify the function works – should be True

def remove_transparent_images_from_csv(path):
    path = Path(path)
    og_csv = pd.read_csv(path, header=0)
    image_paths = np.asarray(og_csv.iloc[:, 1]) # gets second column from the csv (first is indices)
    rows_to_drop = []
    for idx, image_path in enumerate(tqdm(image_paths)):
        if to_remove(image_path):
            rows_to_drop.append(idx)

    df = og_csv.drop(rows_to_drop)
    df = df.reset_index(drop=True)
    df = df.drop(columns=df.columns[0])
    df.to_csv(path.with_stem(path.stem + "_cleaned")) # adds _cleaned to the end of the csv's name

if __name__ == "__main__":
    fire.Fire()
