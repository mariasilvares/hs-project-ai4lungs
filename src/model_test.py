import argparse
import os
import numpy as np
from tqdm import tqdm
import random

# Sklearn Metrics
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

# PyTorch Imports
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

# Project Imports
from model_utilities import ChestXRayNN, OpenCVXRayNN
from dataset_utilities import ChestXRayAbnormalities, OpenCVXray

def main():
    # CLI Arguments
    parser = argparse.ArgumentParser(description='HS-Project-AI4Lungs: Model Testing.')
    parser.add_argument('--gpu_id', type=int, default=0, help="The ID of the GPU to use.")
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('--results_dir', type=str, required=True, help='Path to the results directory.')
    parser.add_argument('--weights_dir', type=str, required=True, help='Directory for loading model weights.')
    parser.add_argument('--history_dir', type=str, required=True, help='Directory for saving test history.')
    parser.add_argument("--data_augmentation", action="store_true", help="Enable data augmentation")
    parser.add_argument('--model_name', type=str, choices=['OpenCVXRayNN', 'ChestXRayNN'], required=True, help='Name of the model to be used')
    parser.add_argument('--dataset_name', type=str, required=True, help='The dataset for the experiments.')
    parser.add_argument('--channels', type=int, default=3, help="Número de canais da imagem.")
    parser.add_argument('--height', type=int, default=64, help="Altura da imagem de entrada.")
    parser.add_argument('--width', type=int, default=64, help="Largura da imagem de entrada.")
    parser.add_argument('--nr_classes', type=int, default=3, help="Número de classes no modelo.")
    parser.add_argument('--batch_size', type=int, default=32, help="Tamanho do lote para o teste.")
    parser.add_argument('--base_data_path', type=str, required=True, help="Caminho base dos dados para teste.")
    
    args = parser.parse_args()

    # Seed de aleatoriedade para reprodutibilidade
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True 

    # Define o dispositivo
    DEVICE = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
    
    # Caminhos para os pesos e modelo a carregar
    weights_dir = args.weights_dir
    model_path = os.path.join(weights_dir, f"{args.model_name.lower()}_val_{args.dataset_name.lower()}.pt")
    
    # Inicializa o modelo conforme o argumento
    if args.model_name == "ChestXRayNN":
        model = ChestXRayNN(channels=args.channels, height=args.height, width=args.width, nr_classes=args.nr_classes)
    elif args.model_name == "OpenCVXRayNN":
        model = OpenCVXRayNN(channels=args.channels, height=args.height, width=args.width, nr_classes=args.nr_classes)
    else:
        raise ValueError(f"Modelo desconhecido: {args.model_name}")
    
    # Define as transformações para teste
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    test_transforms = transforms.Compose([
        transforms.Resize((args.height, args.width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])
    
    # Inicializa o Dataset de Teste
    if args.dataset_name.lower() == "chestxrayabnormalities":
        test_set = ChestXRayAbnormalities(
            base_data_path=args.base_data_path,
            split="test",
            transform=test_transforms
        )
    elif args.dataset_name.lower() == "opencvxray":
        test_set = OpenCVXray(
            base_data_path=args.base_data_path,
            split="test",
            transform=test_transforms
        )
    else:
        raise ValueError(f"Dataset desconhecido: {args.dataset_name}")
    
    # DataLoader
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    
    # Carrega os pesos do modelo
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print(f"Pesos carregados de: {model_path}")
    else:
        raise FileNotFoundError(f"Arquivo de pesos não encontrado: {model_path}")
    
    # Testa o modelo
    print("Iniciando o teste...")
    model.to(DEVICE)
    model.eval()
    
    y_test_true = []
    y_test_pred = []
    run_test_loss = 0.0
    
    LOSS = torch.nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            logits = model(images)
            loss = LOSS(logits, labels)
            run_test_loss += loss.item() * images.size(0)
            probabilities = torch.nn.Softmax(dim=1)(logits)
            predictions = torch.argmax(probabilities, dim=1)
            y_test_true.extend(labels.cpu().numpy())
            y_test_pred.extend(predictions.cpu().numpy())
    
    avg_test_loss = run_test_loss / len(test_loader.dataset)
    test_acc = accuracy_score(y_test_true, y_test_pred)
    test_recall = recall_score(y_test_true, y_test_pred, average='macro')
    test_precision = precision_score(y_test_true, y_test_pred, average='macro')
    test_f1 = f1_score(y_test_true, y_test_pred, average='macro')
    
    print(f"Test Loss: {avg_test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Recall: {test_recall:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test F1-Score: {test_f1:.4f}")
    
    # Salva OS resultados com ou sem data augmentation
    if args.data_augmentation:
        np.save(
            file=os.path.join(args.history_dir, f"{args.model_name.lower()}_ts_{args.dataset_name.lower()}_losses_da.npy"),
            arr=np.array([avg_test_loss]),
            allow_pickle=True
        )
        np.save(
            file=os.path.join(args.history_dir, f"{args.model_name.lower()}_ts_{args.dataset_name.lower()}_metrics_da.npy"),
            arr=np.array([test_acc, test_precision, test_recall, test_f1]),
            allow_pickle=True
        )
    else:
        np.save(
            file=os.path.join(args.history_dir, f"{args.model_name.lower()}_ts_{args.dataset_name.lower()}_losses.npy"),
            arr=np.array([avg_test_loss]),
            allow_pickle=True
        )
        np.save(
            file=os.path.join(args.history_dir, f"{args.model_name.lower()}_ts_{args.dataset_name.lower()}_metrics.npy"),
            arr=np.array([test_acc, test_precision, test_recall, test_f1]),
            allow_pickle=True
        )
    
if __name__ == "__main__":
    main()