# Imports
import os
import _pickle as cPickle
from tqdm import tqdm
import numpy as np
import pandas as pd
from PIL import Image

# Sklearn Imports
from sklearn.model_selection import train_test_split

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
    def __init__(self, base_data_path, split="train", subset='all', transform=None):
        """
        Args:
            base_data_path (string): Data directory.
            split (string): Data split (train, val, test)
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        assert split in ("train", "val", "test")
        assert subset in ('all', 'china', 'montgomery')


        # listas para os caminhos das imagens e dos rótulos
        images_paths = []
        images_labels = []
        images_masks = []
        self.transform = transform
    

        # Subconjuntos de dados para Montgomery e ChinaSet
        if subset == 'all':
            subsets = ["Montgomery", "ChinaSet_AllFiles"]
        elif subset == 'china':
            subsets = ["ChinaSet_AllFiles"]
        else:
            subsets = ["Montgomery"]
        self.subset_ = subset

        for subset in subsets:
            # Caminho para as pastas principais
            main_path = os.path.join(base_data_path, subset)

            if os.path.exists(main_path):
                # Percorrer todas as pastas e subpastas
                for root, dirs, files in os.walk(main_path):
                    # Verificar se a pasta "clinicalreadings" existe dentro do diretório atual
                    if "clinicalreadings" in dirs:
                        clinicalreadings_path = os.path.join(root, "clinicalreadings")
                        for clinical_file in os.listdir(clinicalreadings_path):
                            # Processar arquivos dentro de "clinicalreadings"
                            clinical_file_path = os.path.join(clinicalreadings_path, clinical_file)
                            if clinical_file_path.endswith(".txt"):  # para ler arquivos .txt
                                with open(clinical_file_path, 'r') as file:
                                    print(file.read())  # Para ilustrar como lidar com arquivos de leitura clínica

                    # Processa imagens dentro de "CXR_png", "leftMask", "rightMask"
                    if "CXR_png" in dirs:
                        cxr_path = os.path.join(root, "CXR_png")
                        for img_file in os.listdir(cxr_path):
                            if img_file.endswith(".png"):
                                img_path = os.path.join(cxr_path, img_file)
                                images_paths.append(img_path)

                                # Determina o rótulo da imagem
                                if '0.png' in img_file:
                                    images_labels.append(0)  # Normal
                                elif '1.png' in img_file:
                                    images_labels.append(1)  # Anormal

                    # Verifica e processa "ManualMask" e subpastas "leftMask", "rightMask"
                    if self.subset_ == 'montgomery':
                        if "ManualMask" in dirs:
                            mask_path = os.path.join(root, "ManualMask")
                            for mask_type in ["leftMask", "rightMask"]:
                                mask_subpath = os.path.join(mask_path, mask_type)
                                if os.path.exists(mask_subpath):
                                    for mask_file in os.listdir(mask_subpath):
                                        if mask_file.endswith(".png"):
                                            mask_file_path = os.path.join(mask_subpath, mask_file)
                                            print(f"Máscara encontrada: {mask_file_path}")
                                            images_masks.append(mask_file_path)


        # Perform train, val and test split
        images_total = len(images_paths)
        labels_total = len(images_labels) 
        

        # dividir treino e (validação + teste):
        train_images_paths, test_val_images_paths, train_labels_paths, test_val_labels_paths = train_test_split(images_paths, images_labels, test_size=0.3, random_state=42, stratify=images_labels)
        # dividir validação e teste:
        val_images_paths, test_images_paths, val_labels_paths, test_labels_paths = train_test_split(test_val_images_paths, test_val_labels_paths, test_size=0.5, random_state=42, stratify=test_val_labels_paths)

        
        if split == 'train':
            self.images_paths = train_images_paths
            self.images_labels = train_labels_paths
        elif split == 'val':
            self.images_paths = val_images_paths
            self.images_labels = val_labels_paths
        else:
            self.images_paths = test_images_paths
            self.images_labels = test_labels_paths


        # Class variables
        self.base_data_path = base_data_path
        self.split = split

        # Verificar os tamanhos dos conjuntos
        print(f"Total de imagens para treino: {len(train_images_paths)}")
        print(f"Total de imagens para treino (%): {len(train_images_paths) / images_total * 100}%")
        print(f"Total de imagens para validação: {len(val_images_paths)}")
        print(f"Total de imagens para validação (%): {len(val_images_paths) / images_total * 100}%")
        print(f"Total de imagens para teste: {len(test_images_paths)}")
        print(f"Total de imagens para teste (%): {len(test_images_paths) / images_total * 100}%")
        print(f"Total de imagens no conjunto dividido: {len(train_images_paths) + len(val_images_paths) + len(test_images_paths)}")
        print(f"Total de imagens inicial: {images_total}")

        
            # Verificar os tamanhos dos conjuntos (rótulos)
        print(f"Total de rótulos para treino: {len(train_labels_paths)}")
        print(f"Total de rótulos para treino (%): {len(train_labels_paths) / labels_total * 100}%")
        print(f"Total de rótulos para validação: {len(val_labels_paths)}")
        print(f"Total de rótulos para validação (%): {len(val_labels_paths) / labels_total * 100}%")
        print(f"Total de rótulos para teste: {len(test_labels_paths)}")
        print(f"Total de rótulos para teste (%): {len(test_labels_paths) / labels_total * 100}%")
        print(f"Total de rótulos no conjunto dividido: {len(train_labels_paths) + len(val_labels_paths) + len(test_labels_paths)}")
        print(f"Total de rótulos inicial: {labels_total}")
        
        
        # TODO: Work for the end of the year
        # self.images_masks = images_masks
        self.transform = transform



        # Verificar o número de imagens e rótulos
        print(f"Total de imagens carregadas: {len(self.images_paths)}")
        print(f"Total de rótulos atribuídos: {len(self.images_labels)}")

        assert len(self.images_paths) == len(self.images_labels), "Número de imagens e rótulos não coincide"



    # Method: __len__
    def __len__(self):
        return len(self.images_paths)

    # Method: __getitem__
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Carregar a imagem
        img_path = self.images_paths[idx]
        image = Image.open(img_path).convert("RGB")

        # Carregar o rótulo
        label = self.images_labels[idx]

        # Aplicar transformações (se fornecido)
        if self.transform:
            image = self.transform(image)

        return image, label


# Estatísticas do dataset
if __name__ == "__main__":

    # Calcular estatísticas gerais
    def calculate_statistics(values):
        return np.mean(values), np.min(values), np.max(values)

    CHECK_OpenCVXray = False


    base_data_path = "/home/mariareissilvares/Documents/hs-project-ai4lungs/data/ChestXRayAbnormalities"

    subsets = ["Montgomery/MontgomerySet", "ChinaSet_AllFiles/ChinaSet_AllFiles"]
    for subset in subsets:
        subset_path = os.path.join(base_data_path, subset)
        if not os.path.exists(subset_path):
            print(f"Erro: O diretório {subset_path} não existe!")
        else:
            print("O diretório existe!")

    # Carregar o dataset completo
    dataset = ChestXRayAbnormalities(base_data_path=base_data_path)
    print("Total de imagens:", len(dataset))





    # Testar o split de treino, validação e teste
    if __name__ == "__main__":
        base_data_path = "/home/mariareissilvares/Documents/hs-project-ai4lungs/data/ChestXRayAbnormalities"


        # Subsets a serem testados
    subsets = ["all","montgomery", "china"]

    for subset in subsets:
        print(f"\n ..........Testar o subset: {subset.upper()}..........")

        # Testar o split de treino
        d= ChestXRayAbnormalities(
            base_data_path=base_data_path,
            split="train",
            subset=subset
        )
        print(f"Train size ({subset}): {len(d)}")
        


        # Testar o conjunto de validação
        v = ChestXRayAbnormalities(
            base_data_path=base_data_path,
            split="val",
            subset=subset
        )
        print(f"Validation size ({subset}): {len(v)}")
  
          

        # Testar o conjunto de teste
        t = ChestXRayAbnormalities(
            base_data_path=base_data_path,
            split="test",
            subset=subset
        )
        print(f"Test size ({subset}): {len(t)}")
       


    # Estatísticas de altura, largura e canais
    heights = []
    widths = []
    channels = []

    for idx in range(len(dataset)):
        image, _ = dataset[idx]
        image_np = np.array(image)

        # Extrair dimensões
        heights.append(image_np.shape[0])  # Altura
        widths.append(image_np.shape[1])  # Largura
        channels.append(image_np.shape[2])  # Canais

    

    height_stats = calculate_statistics(heights)
    width_stats = calculate_statistics(widths)
    channel_stats = calculate_statistics(channels)

    print(f"Altura2 - Média: {height_stats[0]:.2f}, Min: {height_stats[1]}, Max: {height_stats[2]}")
    print(f"Largura2 - Média: {width_stats[0]:.2f}, Min: {width_stats[1]}, Max: {width_stats[2]}")
    print(f"Canais2 - Média: {channel_stats[0]:.2f}, Min: {channel_stats[1]}, Max: {channel_stats[2]}")


    
    if CHECK_OpenCVXray:
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
    
  

