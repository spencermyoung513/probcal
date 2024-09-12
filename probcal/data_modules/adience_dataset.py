import os
from pathlib import Path
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# move face_labels into /faces

class AdienceDataset(Dataset):
    def __init__(self, image_dir: str, transform=None):
        self.image_dir = Path(image_dir)

        # read in the labels into a df

        self.transform = transform
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # construct the path to the image from the label[idx]

        img_name = self.image_files[idx]
        img_path = self.image_dir / img_name
        label_path = self.label_dir / f"{img_name.split('.')[0]}.txt"
        
        # TODO :: figure out what form to return the image and label in ?? TensorDataset

        # # Load image
        # image = Image.open(img_path).convert('RGB')
        # if self.transform:
        #     image = self.transform(image)
        
        # # Load label
        # with open(label_path, 'r') as f:
        #     label = f.read().strip()
        
        # # Convert label to tensor (you might need to modify this based on your label format)
        # label = torch.tensor(float(label))
        
        return None, None