#Imports
from django.db import models
from django.contrib.auth.models import User

class XrayImage(models.Model):
    image = models.ImageField(upload_to='xray_images/')  # Onde as imagens serão armazenadas
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Imagem {self.id} carregada em {self.uploaded_at}"