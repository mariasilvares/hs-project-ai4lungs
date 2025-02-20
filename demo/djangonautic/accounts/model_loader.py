import os
import sys

# Subir três níveis para alcançar a raiz do projeto
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
# Adiciona o diretório 'src' ao sys.path
src_path = os.path.join(parent_dir, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

print("sys.path:", sys.path)  # Use para depurar se o caminho foi adicionado corretamente

import torch
from model_utilities import OpenCVXRayNN, ChestXRayNN

# Caminho para os pesos do modelo
MODEL_NAME = 'OpenCVXRayNN'
WEIGHTS_PATH = os.path.join('/home/mariareissilvares/Documents/hs-project-ai4lungs/results/weights/', f'{MODEL_NAME.lower()}_val_opencvxray_da.pt')

if MODEL_NAME == 'OpenCVXRayNN':
    model = OpenCVXRayNN(channels=3, height=64, width=64, nr_classes=3)
elif MODEL_NAME == 'ChestXRayNN':
    model = ChestXRayNN(channels=3, height=64, width=64, nr_classes=3)
else:
    raise ValueError(f"Modelo desconhecido: {MODEL_NAME}")

model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=torch.device('cpu')))
model.eval()