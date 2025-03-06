import torch
from torchvision import transforms
from PIL import Image
from accounts.model_loader import load_model  # Importa a função para carregar o modelo

# Inicializa a variável do modelo como None
model = None

# Transformações de pré-processamento
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

def get_model():
    """
    Retorna a instância do modelo, carregando-o se necessário.
    """
    global model
    if model is None:
        model = load_model()
    return model

def predict_xray(image_path):
    """
    Realiza a predição de uma imagem de raio-X utilizando o modelo carregado.
    """
    try:
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0)

        model = get_model()
        with torch.no_grad():
            output = model(image)
            prediction = torch.argmax(output, dim=1).item()

        return prediction
    except Exception as e:
        return f"Erro na predição: {e}"