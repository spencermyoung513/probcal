import os
import torch
import pandas as pd
import numpy as np
from skimage import io, transform
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class AAFDataset(Dataset):
    """The All-Age-Faces (AAF) Dataset contains 13'322 face images (mostly Asian) 
    distributed across all ages (from 2 to 80), including 7381 females and 5941 males."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.pictures_df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.pictures_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.pictures_df.iloc[idx, 0])
        image = io.imread(img_name)
        age = self.pictures_df.iloc[idx, 1]
        gender = self.pictures_df.iloc[idx, 2]
        
        sample = {'image': image, 'age': age, 'gender': gender}

        if self.transform:
            sample = self.transform(sample)

        return sample
    

face_dataset = AAFDataset(csv_file='data/aaf/image sets/picture_data.csv',
                                    root_dir='data/aaf/original images')

fig = plt.figure()

for i, sample in enumerate(face_dataset):
    print(i, sample['image'].shape, sample['age'], sample['gender'])

    ax = plt.subplot(1, 4, i + 1)
    plt.tight_layout()
    ax.set_title('Sample #{}'.format(i))
    ax.imshow(sample['image'])


    if i == 3:
        plt.show()
        break