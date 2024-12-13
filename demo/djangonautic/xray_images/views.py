from django.shortcuts import render
from .forms import XrayImageForm
from .utils import process_image  # Função para processar a imagem após upload

def upload_image(request):
    if request.method == 'POST' and request.FILES['image']:
        form = XrayImageForm(request.POST, request.FILES)
        if form.is_valid():
            # Salva a imagem no banco de dados
            xray_image = form.save()
            
            # Processa a imagem usando a função de processamento
            result = process_image(xray_image.image.path)
            
            # Exibe o resultado da análise
            return render(request, 'xray_images/upload_image.html', {'form': form, 'message': f'Resultado da análise: {result}'})
    else:
        form = XrayImageForm()
    return render(request, 'xray_images/upload_image.html', {'form': form})
