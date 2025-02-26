import torch
from torchvision import transforms
from PIL import Image
from accounts.model_loader import load_model  # Agora importa a função, não o modelo

# Carrega o modelo apenas quando necessário
model = load_model()

# Transformações de pré-processamento
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

def predict_xray(image_path):
    """
    Realiza a predição de uma imagem de raio-X utilizando o modelo carregado.
    """
    try:
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = model(image)
            prediction = torch.argmax(output, dim=1).item()

        return prediction
    except Exception as e:
        return f"Erro na predição: {e}"