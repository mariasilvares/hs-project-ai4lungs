from django import forms
from django.contrib.auth.models import User

class EditProfileForm(forms.ModelForm):
    profile_picture = forms.ImageField(required=False)  # Para permitir upload de imagem

    class Meta:
        model = User
        fields = ['username', 'email']  # Campos que serão editáveis pelo usuário

