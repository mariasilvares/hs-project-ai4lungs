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
        
        # Carrega os caminhos das imagens e os rótulos com base no conjunto de dados (train, val, test)
        split_folder = os.path.join(base_data_path, split)
        # print(split_folder)
        # print(os.path.exists(split_folder))
        
        # TODO
        self.data=pd.read_csv(labels_file) #lê o CSv e armazena as informações num DataFrame
        
        # Extrai as colunas do DatafRame para listas, onde images_paths contém os caminhos das imagens e images_labels contém os rótulos
        self.images_paths = self.data["images_path"].tolist() #lista de caminhos das imagens
        self.images_labels = self.data["label"].tolist() #lista de rótulos das imagens


        # Class variables
        self.base_data_path = base_data_path
        self.split = split
        self.images_paths = images_path
        self.images_labels = images_label
        self.transform = transform


        return
    


    # Method: __len__
    def __len__(self):
        #  Your code here
        return len(self.images_paths) # retorna o nº total de amostras (imagens) no dataset


    # Method: __getitem__
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        

        # Get images
        # Your code here
        img_path =os.paths.join(self.base_data_path, self.images_paths[idx]) #cria o caminho completo para a imagem e o caminho específico da imagem 
        image=Image.open(img_path).convert("RGB")#abre a imagem e converte-a para o formato RGB

        # Get labels
        # Your code here
        label=self.images_labels[idx] #pega no rótulo correspondente ao indice da imagem

        # Apply transformation
        if self.transform:
            image = self.transform(image)

        return image, label

 

if __name__ == "__main__":
    d = OpenCVXray(
        base_data_path="/home/mariareissilvares/Documents/hs-project-ai4lungs/data/OpenCVXray",
        split="train"   
    )

    print(len(d))