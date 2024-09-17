import os
from pathlib import Path
import torch
import pandas as pd
# from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms, utils
import tarfile
import requests
from tqdm import tqdm

class AdienceDataset(Dataset):
    """Adience dataset """
    def __init__(self, root_dir: str, transform=None):
        """
        Arguments:
            image_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = Path(root_dir)
        self.image_dir = self.root_dir.joinpath("faces")
        self.labels_dir = self.image_dir.joinpath("face_labels")

        self._download_images()
        self._download_labels()

        # read the labels in from {image_dir}/face_labels dir
        labelFilePaths = [f for f in os.listdir(self.labels_dir)]
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
    
    def _already_downloaded(self) -> bool:
        return (
            self.labels_dir.exists()
            and self.image_dir.exists()
            and any(self.image_dir.iterdir())
            and any(self.labels_dir.iterdir())
        )
    
    def _download_images(self):
        if self._already_downloaded():
            return
        archive_file = "faces.tar.gz"

        if self.root_dir.joinpath(archive_file).exists():
            self._unzip_data()
            return

        self._download_file(archive_file, self.root_dir)
        self._unzip_data()
        

    def _download_labels(self):
        if not self.labels_dir.exists():
            os.mkdir(self.labels_dir)
        if self._already_downloaded():
            return
        files = ["fold_0_data.txt", "fold_1_data.txt", "fold_2_data.txt", "fold_3_data.txt", "fold_4_data.txt"]

        for file in files:
            self._download_file(file, self.labels_dir)

    def _download_file(self, filename, dir):
        base_url = "http://www.cslab.openu.ac.il/download/adiencedb/AdienceBenchmarkOfUnfilteredFacesForGenderAndAgeClassification/"
        username = "adiencedb"
        password = "adience"

        # Send a GET request to the URL with authentication
        response = requests.get(base_url+filename, auth=(username, password), stream=True)
        
        # Check if the request was successful
        if response.status_code == 200:
            # Get the total file size
            total_size = int(response.headers.get('content-length', 0))
            # Open the output file and write the content
            with open(dir.joinpath(filename), 'wb') as file, tqdm(
                desc=filename,
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as progress_bar:
                for data in response.iter_content(chunk_size=1024):
                    size = file.write(data)
                    progress_bar.update(size)
            print(f"File downloaded successfully: {filename}")
        else:
            print(f"Failed to download file. Status code: {response.status_code}")

    def _unzip_data(self):
        # open file 
        file = tarfile.open(self.root_dir.joinpath("faces.tar.gz")) 
        # extracting file 
        file.extractall(self.root_dir) 
        file.close() 
        
dataset = AdienceDataset("../../data")
print(len(dataset))
print(len(dataset.labelsDF))
print(dataset[0])

df = dataset.labelsDF

# import ast
# from collections import Counter

# def analyze_age_data(df, column_name):
#     def is_range(value):
#         try:
#             return isinstance(ast.literal_eval(value), tuple)
#         except:
#             return False

#     scalar_count = 0
#     range_count = 0
#     range_counter = Counter()

#     for value in df[column_name]:
#         if isinstance(value, str):
#             if is_range(value):
#                 range_count += 1
#                 # Convert string representation of tuple to actual tuple
#                 tuple_range = ast.literal_eval(value)
#                 range_counter[tuple_range] += 1
#             else:
#                 try:
#                     float(value)
#                     scalar_count += 1
#                 except ValueError:
#                     pass
#         elif isinstance(value, (int, float)):
#             scalar_count += 1

#     return scalar_count, range_count, range_counter

# scalar_count, range_count, range_counter = analyze_age_data(df, 'age')

# print(f"Number of scalar values: {scalar_count}")
# print(f"Number of range values: {range_count}")

# total = scalar_count + range_count
# scalar_percentage = (scalar_count / total) * 100
# range_percentage = (range_count / total) * 100

# print(f"\nPercentage of scalar values: {scalar_percentage:.2f}%")
# print(f"Percentage of range values: {range_percentage:.2f}%")

# print("\nFrequency of each tuple range:")
# for range_tuple, count in range_counter.most_common():
#     print(f"{range_tuple}: {count}")

# # Optional: Calculate percentage of each range within range values
# if range_count > 0:
#     print("\nPercentage of each range within range values:")
#     for range_tuple, count in range_counter.most_common():
#         percentage = (count / range_count) * 100
#         print(f"{range_tuple}: {percentage:.2f}%")