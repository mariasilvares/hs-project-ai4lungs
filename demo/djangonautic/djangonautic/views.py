from django.http import HttpResponse
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from .forms import EditProfileForm
from django.contrib import messages

def homepage (request):
   # return HttpResponse('homepage')
   return render (request,'homepage.html')


def profile(request):
    # Lógica para exibir o perfil
    return render(request, 'profile.html')

def profile_edit(request):
    # Lógica para editar o perfil
    return render(request, 'profile_edit.html')





