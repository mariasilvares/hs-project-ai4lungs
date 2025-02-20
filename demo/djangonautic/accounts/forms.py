from django import forms
from django.contrib.auth.forms import UserChangeForm
from django.contrib.auth.models import User
from .models import UserProfile
from .models import Patient, MedicalImage

class UserEditForm(UserChangeForm):
    class Meta:
        model = User
        fields = ['username', 'email']  # Campos para o usuário editar

class UserProfileForm(forms.ModelForm):
    class Meta:
        model = UserProfile
        fields = ['profile_picture']  # Campo para editar a foto de perfil

class PatientForm(forms.ModelForm):
    class Meta:
        model = Patient
        fields = ['name', 'number']
        labels = {
            'name': 'Nome',
            'number': 'Número',
        }

class MedicalImageForm(forms.ModelForm):
    class Meta:
        model = MedicalImage
        fields = ['image', 'description']

