# inference.py
import torch
from torchvision import transforms
from PIL import Image
from .model_loader import model

def predict(image_path):
    # Define as transformações necessárias
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Abre a imagem
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0)  # Adiciona uma dimensão extra para o batch

    # Realiza a inferência
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    # Mapeia a predição para a classe correspondente
    classes = ['Covid', 'Pneumonia', 'Normal'] 
    return classes[predicted.item()]
