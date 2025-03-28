import torch
import os
import sys

# Caminho para o diretório do projeto
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, parent_dir)

# Caminho para os pesos do modelo
MODEL_NAME = 'DenseNet121OpenCVXRayNN'
WEIGHTS_PATH = os.path.join(parent_dir, 'results/weights', f'{MODEL_NAME.lower()}_val_opencvxray_da.pt')

# Função para carregar o modelo 
def load_model():
    from src.model_utilities import DenseNet121OpenCVXRayNN
    model = DenseNet121OpenCVXRayNN(channels=3, height=64, width=64, nr_classes=3)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=torch.device('cpu')))
    model.eval()
    return model

# Instancia o modelo apenas quando necessário
model = load_model()