# Imports
import os
import numpy as np
from tqdm import tqdm

# Sklearn Metrics
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

# PyTorch Imports
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

# Project Imports
from model_utilities import ChestXRayNN, OpenCVXRayNN
from dataset_utilities import ChestXRayAbnormalities, OpenCVXray

# Choose GPU
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Configurations
MODEL_NAME = "ChestXRayNN" 
DATASET_NAME = "ChestXRayAbnormalities"  
BATCH_SIZE = 32

# Paths
weights_dir = os.path.join("results", DATASET_NAME, "weights")
model_path = os.path.join(weights_dir, f"{MODEL_NAME.lower()}_val_{DATASET_NAME.lower()}.pt")

# Inicializa o modelo
if MODEL_NAME == "ChestXRayNN":
    model = ChestXRayNN(channels=3, height=64, width=64, nr_classes=3)
elif MODEL_NAME == "OpenCVXRayNN":
    model = OpenCVXRayNN(channels=3, height=64, width=64, nr_classes=2)
else:
    raise ValueError(f"Modelo desconhecido: {MODEL_NAME}")

# Mean and STD to Normalize
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
test_transforms = transforms.Compose([
        transforms.Resize((64, 64)),  
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ]) 

# Inicializa o dataset
if DATASET_NAME == "ChestXRayAbnormalities":
    test_set = ChestXRayAbnormalities(
        base_data_path="/home/mariareissilvares/Documents/hs-project-ai4lungs/data/ChestXRayAbnormalities",
        split="test",
        transform=test_transforms
    )
elif DATASET_NAME == "OpenCVXray":
    test_set = OpenCVXray(
        base_data_path="/home/mariareissilvares/Documents/hs-project-ai4lungs/data/OpenCVXray",
        split="test",
        transform=test_transforms
    )
else:
    raise ValueError(f"Dataset desconhecido: {DATASET_NAME}")

# DataLoader
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

# Carrega os pesos do modelo
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    print(f"Pesos carregados de: {model_path}")
else:
    raise FileNotFoundError(f"Arquivo de pesos não encontrado: {model_path}")

# Testa do modelo
print("Iniciando o teste...")
model.to(DEVICE)
model.eval()

# Inicializa as métricas
y_test_true = []
y_test_pred = []
run_test_loss = 0.0

LOSS = torch.nn.CrossEntropyLoss()  

# Loop de Teste
with torch.no_grad():
    for images, labels in tqdm(test_loader):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        # Forward pass
        logits = model(images)
        loss = LOSS(logits, labels)
        run_test_loss += (loss.item() * images.size(0))

        # Predição
        probabilities = torch.nn.Softmax(dim=1)(logits)
        predictions = torch.argmax(probabilities, dim=1)

        y_test_true.extend(labels.cpu().numpy())
        y_test_pred.extend(predictions.cpu().numpy())

# Cálculo das métricas
avg_test_loss = run_test_loss / len(test_loader.dataset)
test_acc = accuracy_score(y_test_true, y_test_pred)
test_recall = recall_score(y_test_true, y_test_pred, average='macro')
test_precision = precision_score(y_test_true, y_test_pred, average='macro')
test_f1 = f1_score(y_test_true, y_test_pred, average='macro')


# Resultados
print(f"Test Loss: {avg_test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test Recall: {test_recall:.4f}")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test F1-Score: {test_f1:.4f}")

