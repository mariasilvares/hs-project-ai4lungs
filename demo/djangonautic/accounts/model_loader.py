import os
import torch
from model_utilities import OpenCVXRayNN, ChestXRayNN

# caminho para os pesos do modelo
MODEL_NAME = 'OpenCVXRayNN' 
WEIGHTS_PATH = os.path.join('/home/mariareissilvares/Documents/hs-project-ai4lungs/results/weights/', f'{MODEL_NAME.lower()}_val_opencvxray_da.pt')

# Inicializa o modelo
if MODEL_NAME == 'OpenCVXRayNN':
    model = OpenCVXRayNN(channels=3, height=64, width=64, nr_classes=3)
elif MODEL_NAME == 'ChestXRayNN':
    model = ChestXRayNN(channels=3, height=64, width=64, nr_classes=3)
else:
    raise ValueError(f"Modelo desconhecido: {MODEL_NAME}")

# Carrega os pesos do modelo
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=torch.device('cpu')))
model.eval()  # Coloca o modelo em modo de avaliação