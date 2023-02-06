import os
import numpy as np
import pytorch_lightning as pl
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
import matplotlib.pyplot as plt

from tqdm import tqdm
from PIL import Image
from torchvision import transforms

CSV = "/deep/u/ayushsn/satellite-pixel-synthesis-pytorch/road_classification/train.csv"

# Function for setting the seed
pl.seed_everything(42)

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.determinstic = True
torch.backends.cudnn.benchmark = False

class RoadDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path=CSV):
        self.df = pd.read_csv(csv_path)
        self.df_labels=self.df[['has_road']]
        self.labels=torch.tensor(self.df_labels.to_numpy().reshape(-1)).long()
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4300, 0.3860, 0.3388), (0.1870, 0.1533, 0.1267), inplace=True),
            ]
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        filename = self.df.at[self.df.index[idx], 'image']
        image = Image.open(filename).convert('RGB')
        image = self.transform(image)
        return image, self.labels[idx]

def get_dataset_mean_std():
    """
    Result: mean: tensor([0.4300, 0.3860, 0.3388]); std:  tensor([0.1870, 0.1533, 0.1267])
    """
    dataset = RoadDataset(csv_path=CSV)
    # data loader
    image_loader = torch.utils.data.DataLoader(dataset, 
                              batch_size  = 8, 
                              shuffle     = False, 
                              num_workers = 0,
                              pin_memory  = True)
    
    # batch shape: torch.Size([b, c, h, w])
    
    psum    = torch.tensor([0.0, 0.0, 0.0])
    psum_sq = torch.tensor([0.0, 0.0, 0.0])

    # loop through images
    for inputs in tqdm(image_loader):
        psum    += inputs[0].sum(axis        = [0, 2, 3])
        psum_sq += (inputs[0] ** 2).sum(axis = [0, 2, 3])
    
    count = len(dataset) * 500 * 500

    # mean and std
    total_mean = psum / count
    total_var  = (psum_sq / count) - (total_mean ** 2)
    total_std  = torch.sqrt(total_var)

    # output
    print('mean: '  + str(total_mean))
    print('std:  '  + str(total_std))
