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


                    # Convert labels (string) to unique ints
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

        return image, label



# TODO: ChestXRayAbnormalities: Dataset Class
class ChestXRayAbnormalities(Dataset):

    # Method: __init__
    def __init__(self, base_data_path, subset='train', transform=None):
        """
        Args:s
            base_data_path (string): Data directory.
            split (string): Data split (train, val, test)
            transform (callable, optional): Optional transform to be applied on a sample.
        """

        assert subset in ("train", "val", "test")
        



        # Initialise variables
        
        subset_path = os.path.join(base_data_path, subset) # Caminho completo para a pasta do conjunto de dados (train, val, test)
        # print(split_folder)
        # print(os.path.exists(split_folder))


        # listas para os caminhos das imagens e dos rótulos
        images_paths=[]
        images_labels=[]


        # Subpastas que representam classes
        for label_name in ['abnormal', 'normal']:
            label_path = os.path.join(subset_path, label_name)
            if os.path.exists(label_path):
                images = os.listdir(label_path)  
                for img_file in images:
                    img_path = os.path.join(label_path, img_file)
                    images_paths.append(img_path)
                    
            
                    # Convert labels (string) to unique ints
                    if label_name == 'abnormal':
                        label_name = 0
                    elif label_name == 'normal':
                        label_name = 1

                images_labels.append(label)

            return len(images_paths) == len(images_labels)


        # Class variables
        self.base_data_path = base_data_path
        self.subset = subset
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

        return image, label
    
if __name__ == "__main__":
    base_path= "/home/mariareissilvares/Documents/hs-project-ai4lungs/data/ChestXRayAbnormalities"


    train_dataset = ChestXRayAbnormalities(base_data_path=base_path, subset='train')
    val_dataset = ChestXRayAbnormalities(base_data_path=base_path, subset='val')
    test_dataset = ChestXRayAbnormalities(base_data_path=base_path, subset='test')

    print('Train size2:', len(train_dataset))
    print('val size2:', len(val_dataset))
    print('test size2:', len(test_dataset))

        
    # Estatísticas para altura, largura e canais
    channel_values = {"abnormal": [], "normal": []}
    height_values = {"abnormal": [], "normal": []}
    width_values = {"abnormal": [], "normal": []}

    for subset_name in ["train", "val", "test"]:
        dataset = ChestXRayAbnormalities(base_data_path=base_path, subset=subset_name)
        print(f"{subset_name.capitalize()} size:", len(dataset))

        for idx in range(len(dataset)):
            image, label = dataset[idx]
            label_name = ["abnormal", "normal"][label]

            # Converter imagem para numpy
            image_np = np.array(image)
            channel_values[label_name].append(image_np.shape[2])
            height_values[label_name].append(image_np.shape[0])
            width_values[label_name].append(image_np.shape[1])

    # Estatísticas gerais
    def calculate_statistics(values_dict, label_names):
        total_values = []
        for label in label_names:
            total_values.extend(values_dict[label])

        mean_value = np.mean(total_values)
        min_value = np.min(total_values)
        max_value = np.max(total_values)

        return mean_value, min_value, max_value

    labels = ["abnormal", "normal"]

    channel_stats = calculate_statistics(channel_values, labels)
    height_stats = calculate_statistics(height_values, labels)
    width_stats = calculate_statistics(width_values, labels)

    print(f"Channels - Mean: {channel_stats[0]:.4f}, Min: {channel_stats[1]}, Max: {channel_stats[2]}")
    print(f"Height - Mean: {height_stats[0]:.4f}, Min: {height_stats[1]}, Max: {height_stats[2]}")
    print(f"Width - Mean: {width_stats[0]:.4f}, Min: {width_stats[1]}, Max: {width_stats[2]}")


    
 


if __name__ == "__main__":
    
    # train
    d = OpenCVXray(
        base_data_path="/home/mariareissilvares/Documents/hs-project-ai4lungs/data/OpenCVXray",
        split="train"   
    )

    print("Train size: ", len(d))

    # for idx in range(len(d)):
    #     d_image, d_label = d.__getitem__(idx)
    #     print(np.array(d_image).shape)
    #     print(d_label)



    # val
    v = OpenCVXray(
        base_data_path="/home/mariareissilvares/Documents/hs-project-ai4lungs/data/OpenCVXray",
        split="val"
    )   

    print("Val size: ", len(v))

    # for idx in range(len(v)):
    #     v_image,v_label= v.__getitem__(idx)
    #     print (np.array(v_image).shape) 
    #     print (v_label)



    # test
    t = OpenCVXray(
        base_data_path="/home/mariareissilvares/Documents/hs-project-ai4lungs/data/OpenCVXray",
        split="test"   
    )

    print("Test size: ", len(t))

    # for idx in range(len(t)):
    #     t_image, t_label = t.__getitem__(idx)
    #     print(np.array(t_image).shape)
    #     print(t_label)



    # Inicializar listas para armazenar as estatísticas 
        
    channel_values= {"covid": [], "pneumonia": [], "normal":[]}
    height_values= {"covid": [], "pneumonia": [], "normal":[]}
    width_values= {"covid": [], "pneumonia": [], "normal":[]}

    # Percorre o dataset e armazena valores com base no rótulo
    # Get the values of train, val and test (currently we only have training)
    
    for split in ["train","val","test"]:
        dataset = OpenCVXray(
        base_data_path="/home/mariareissilvares/Documents/hs-project-ai4lungs/data/OpenCVXray",
        split= split
        )
        print(f"{split.capitalize()}size:", len(dataset))  

        for idx in range(len(dataset)):
            image, label = dataset.__getitem__(idx)
            label_name = ["covid", "pneumonia", "normal"][label] # Mapeia o rótulo numérico para o nome
                
            # Converte imagem para numpy e armazena nos dicionários
            image_np= np.array(image)
            channel_values[label_name].append(image_np.shape[0]) # Número de canais por imagem
            height_values[label_name].append(image_np.shape[1]) # Altura por imagem
            width_values[label_name].append(image_np.shape[2]) # Largura por imagem



    # Calcula e imprime as estatísticas para cada rótulo
    channel_values_total = []
    height_values_total = []
    width_values_total = []

    # Consolidar todas as listas em listas totais
    for label_name in ["covid", "pneumonia", "normal"]:
        channel_values_total.extend(channel_values[label_name])
        height_values_total.extend(height_values[label_name])
        width_values_total.extend(width_values[label_name])


    # Update this block to receive the total(s)
    print(f"Canal - Média: {np.mean(channel_values_total):.4f},"
            f" Mínimo: {np.min(channel_values_total):.4f},"
            f" Máximo: {np.max(channel_values_total):.4f}")
            
    print(f"Altura - Média: {np.mean(height_values_total):.4f},"
            f" Mínimo: {np.min(height_values_total):.4f},"
            f" Máximo: {np.max(height_values_total):.4f}")
            
    print(f"Largura - Média: {np.mean(width_values_total):.4f},"
            f" Mínimo: {np.min(width_values_total):.4f},"
            f" Máximo: {np.max(width_values_total):.4f}")
    
  

