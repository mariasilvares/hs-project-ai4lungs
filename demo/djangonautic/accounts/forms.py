# forms.py
from django import forms
from django.contrib.auth.forms import UserChangeForm
from django.contrib.auth.models import User
from .models import UserProfile

class UserEditForm(UserChangeForm):
    class Meta:
        model = User
        fields = ['username', 'email']  # Campos que você quer que o usuário possa editar

class UserProfileForm(forms.ModelForm):
    class Meta:
        model = UserProfile
        fields = ['profile_picture']  # Campo para editar a foto de perfil



