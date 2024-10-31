# Imports
import os
import _pickle as cPickle
from tqdm import tqdm
import numpy as np
import pandas as pd
from PIL import Image

# PyTorch Imports
import torch
from torch.utils.data import Dataset



# OpenCVXray: Dataset Class
class OpenCVXray(Dataset):

    # Method: __init__
    def __init__(self, base_data_path, split="train", transform=None):
        """
        Args:
            base_data_path (string): Data directory.
            split (string): Data split (train, val, test)
            transform (callable, optional): Optional transform to be applied on a sample.
        """

        assert split in ("train", "val", "test")
        

        # Initialise variables
        # Your code here


        # Class variables
        self.base_data_path = base_data_path
        self.split = split
        self.images_paths = images_paths
        self.images_labels = images_labels
        self.transform = transform


        return


    # Method: __len__
    def __len__(self):
        #  Your code here
        pass


    # Method: __getitem__
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        

        # Get images
        # Your code here

        # Get labels
        # Your code here

        # Apply transformation
        if self.transform:
            image = self.transform(image)

        return image, label
