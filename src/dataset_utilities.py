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
from torchvision.transforms import v2



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
        
        split_folder = os.path.join(base_data_path, split) # Caminho completo para a pasta do conjunto de dados (train, val, test)
        # print(split_folder)
        # print(os.path.exists(split_folder))


        # listas para os caminhos das imagens e dos rótulos
        images_paths=[]
        images_labels=[]


        # Listar as subpastas "covid","pneumonia" e "normal" para carregar imagens e rótulos
        for label in ["covid","pneumonia","normal"]:
            folder_path = os.path.join(split_folder,label) # Caminho completo da subclasse para cada classe 
        
            if os.path.exists(folder_path): # Verifica se a pasta da classe existe 
                images = os.listdir(folder_path)
                for img_name in images:
                    img_path = os.path.join(folder_path,img_name) # Caminho completo para cada imagem
                    images_paths.append(img_path) # Adiciona o caminho da imagen à lista de caminhos


                    # TODO: Convert labels (string) to unique ints
                    if label == "covid":
                        label= 0
                    elif label == "pneumonia":
                        label= 1
                    elif label == "normal":
                        label= 2    
            

                    images_labels.append(label) # Adiciona o rótulo da imagem à lista de rótulos
        
        assert len(images_paths) == len(images_labels)



        # Class variables
        self.base_data_path = base_data_path
        self.split = split
        self.images_paths = images_paths
        self.images_labels = images_labels
        self.transform = transform


        return
    


    # Method: __len__
    def __len__(self):
        return len(self.images_paths) # Retorna o número total de amostras (imagens) no dataset


    # Method: __getitem__
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        

        # Get images
        img_path = self.images_paths[idx] # Caminho completo da imagem
        image=Image.open(img_path).convert("RGB") # Abre a imagem e converte-a para o formato RGB

        # Get labels
        label=self.images_labels[idx] # Rótulo correspondente ao indice da imagem

        # Apply transformation (if we have transforms)
        if self.transform:
            image = self.transform(image)

        # If we don't have transforms, convert image into torch.Tensor
        else:
            # Convert PIL to Tensor
            image = v2.PILToTensor()(image)

            # Normalize expects float input
            image = v2.ToDtype(torch.float32, scale=True)(image)

            # Normalise this tensor mean=0.5, std=0.5
            image = v2.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5])(image)
            # print(image)

        return image, label

 

if __name__ == "__main__":
    
    # train
    d = OpenCVXray(
        base_data_path="/home/mariareissilvares/Documents/hs-project-ai4lungs/data/OpenCVXray",
        split="train"   
    )

    print("Train size: ", len(d))

    for idx in range(len(d)):
        d_image, d_label = d.__getitem__(idx)
        print(np.array(d_image).shape)
        print(d_label)



    # val
    v = OpenCVXray(
        base_data_path="/home/mariareissilvares/Documents/hs-project-ai4lungs/data/OpenCVXray",
        split="val"   
    )

    print("Val size: ", len(v))

    for idx in range(len(v)):
        v_image,v_label= v.__getitem__(idx)
        print (np.array(v_image).shape) 
        print (v_label)



    # test
    t = OpenCVXray(
        base_data_path="/home/mariareissilvares/Documents/hs-project-ai4lungs/data/OpenCVXray",
        split="test"   
    )

    print("Test size: ", len(t))

    for idx in range(len(t)):
        t_image, t_label = t.__getitem__(idx)
        print(np.array(t_image).shape)
        print(t_label)