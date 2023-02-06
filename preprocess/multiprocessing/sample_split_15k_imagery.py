from pathlib import Path
from tqdm import tqdm
import pandas as pd
import random

IMAGERY_PATH = Path("/deep/group/aicc-bootcamp/transportation/data/dvrpc/aerial/imagery")

if __name__ == "__main__":
    random.seed(3407)
    print("Populating imagery list...")
    imagery = list(IMAGERY_PATH.glob('*.json'))
    print("Populated imagery list.")
    print("Sampling imagery...")
    sampled_imagery = random.sample(imagery, 15000)
    print("Sampled imagery.")
    print("Creating CSVs...")
    m = 0
    for n in tqdm([3000, 6000, 9000, 12000, 15000]):
        series = pd.Series(sampled_imagery[m:n])
        series.to_csv(f'/deep/u/ayushsn/satellite-pixel-synthesis-pytorch/preprocess/train_road_{n}.csv')
        m = n
    print("Created CSVs.")
