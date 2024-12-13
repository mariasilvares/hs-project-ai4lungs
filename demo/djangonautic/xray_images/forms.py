from django import forms
from .models import XrayImage

class XrayImageForm(forms.ModelForm):
    class Meta:
        model = XrayImage
        fields = ['image']