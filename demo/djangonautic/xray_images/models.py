from django.db import models
from django.contrib.auth import get_user_model

class XrayImage(models.Model):
    image = models.ImageField(upload_to='xray_images/')
    # user = models.ForeignKey('auth.User', on_delete=models.CASCADE)
    user = models.ForeignKey(get_user_model(), on_delete=models.CASCADE, default=None)  # Definindo o valor padrão como None
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Image uploaded by {self.user.username} on {self.uploaded_at}"

