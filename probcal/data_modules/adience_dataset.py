import os
from pathlib import Path
import torch
import pandas as pd
# from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms, utils

class AdienceDataset(Dataset):
    def __init__(self, image_dir: str, transform=None):
        self.image_dir = Path(image_dir)
        labelFilePaths = [f for f in os.listdir(self.image_dir.joinpath("face_labels"))]
        labels = pd.DataFrame()
        for file in labelFilePaths:
            temp = pd.read_csv(self.image_dir.joinpath("face_labels", file), sep='\t')
            labels = pd.concat([labels,temp])
        self.labelsDF = labels
        self.transform = transform

        self.image_files = []
        for dir in os.listdir(self.image_dir):
            if os.path.isdir(self.image_dir.joinpath(dir)):
                self.image_files.extend([file for file in os.listdir(self.image_dir.joinpath(dir)) if file.endswith(('.jpg','.jpeg','.png'))])
        
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        file_name = f"cours_tilt_aligned_face.{self.labelsDF.iloc[idx]['face_id']}.{self.labelsDF.iloc[idx]['original_image']}"
        img_path = self.image_dir.joinpath(self.labelsDF.iloc[idx]['user_id'], file_name)
        print(img_path)
        
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
        
        # FIXME :: the label will be a tensor, this is just a placeholder for now
        # TODO :: some of the age labels are a tuple with a range (e.g. (60, 100)). What should the data be output as
        return None, self.labelsDF.iloc[idx]['age']