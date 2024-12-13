from PIL import Image
import numpy as np

def process_image(image_path):
    """
    Processa a imagem para identificar anomalias.
    Neste exemplo, converte a imagem para escala de cinza.
    """
    image = Image.open(image_path)
    image = image.convert('L')  # Convertendo para escala de cinza, por exemplo
    image_array = np.array(image)
    
    # analisar a imagem

    result = "Não detetado cancro"  # Exemplo de resultado
    return result


