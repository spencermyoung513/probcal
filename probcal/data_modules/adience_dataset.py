import os
from pathlib import Path
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class AdienceDataset(Dataset):
    def __init__(self, image_dir: str, transform=None):
        self.image_dir = Path(image_dir)
        labelFilePaths = [f for f in os.listdir(image_dir/"face_labels") if f.endswith('.txt')]
        labels = pd.DataFrame()
        for file in labelFilePaths:
            temp = pd.read_csv(image_dir/"face_labels"/file, sep='\t')
            labels = labels.append(temp)
        self.labelsDF = labels
        self.transform = transform

        # TODO: Im not sure if this is fully correct
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = f"{self.image_dir}/{self.labelsDF[idx]['user_id']}/cours_tilt_aligned_face.{self.labelsDF[idx]['face_id']}.{self.labelsDF[idx]['original_image']}"
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