from django.shortcuts import render, redirect
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth import login, logout
from django.contrib.auth.decorators import login_required
from .forms import UserEditForm, UserProfileForm
from .models import Activity
from django.contrib import messages
from django.utils.timezone import now
from .models import Activity, UserProfile
from django.core.exceptions import ValidationError


# Create your views here.
def signup_view(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            # log the user in
            login(request, user)
            return redirect('articles:list')
    else:
        form = UserCreationForm()
    return render(request, 'accounts/signup.html', {'form':form})


def login_view(request):
    if request.method == 'POST':
        form = AuthenticationForm(data=request.POST)
        if form.is_valid():
            # log in the user
            user = form.get_user()
            login(request, user)
            return redirect('home')
    else:
        form = AuthenticationForm()
    return render(request, 'accounts/login.html', {'form':form})


def logout_view(request):
    if request.method == 'POST':
        logout(request)
        return redirect('home')


@login_required
def profile_view(request):
    return render(request, 'accounts/profile.html')
    

def profile_edit(request):
    if request.method == 'POST':
        user_form = UserEditForm(request.POST, instance=request.user)
        profile_form = UserProfileForm(request.POST, request.FILES, instance=request.user.userprofile)
        
        if user_form.is_valid() and profile_form.is_valid():
            # Salva o formulário de usuário
            user_form.save()

            # Verifica se o perfil foi alterado (imagem do perfil)
            if profile_form.has_changed():  # Verifica se houve mudanças no perfil
                profile_form.save()

                # Registrar atividade de upload de imagem, se houver
                if 'profile_picture' in profile_form.cleaned_data and profile_form.cleaned_data['profile_picture']:
                    Activity.objects.create(
                        user=request.user,
                        action='image_upload',
                        additional_info=f"Imagem carregada: {profile_form.cleaned_data['profile_picture'].name}"
                    )

            # Registrar a atividade de alteração de perfil
            Activity.objects.create(
                user=request.user,
                action='profile_update',
                additional_info="Alteração no perfil do usuário."
            )

            messages.success(request, "Perfil atualizado com sucesso.")
            return redirect('profile')  # Redireciona para o perfil após salvar
        else:
            messages.error(request, "Ocorreu um erro ao atualizar o perfil. Verifique os campos e tente novamente.")
    else:
        user_form = UserEditForm(instance=request.user)
        profile_form = UserProfileForm(instance=request.user.userprofile)
    
    return render(request, 'accounts/profile_edit.html', {
        'user_form': user_form,
        'profile_form': profile_form
    })

def upload_image(request):
    if request.method == 'POST' and request.FILES['image']:
        #  processando a imagem aqui
        image = request.FILES['image']
        
        # processamento de imagem
        
        # Sucesso ao carregar a imagem
        messages.success(request, 'Imagem carregada com sucesso!')
        
        # Redireciona para uma outra página ou para o próprio upload
        return redirect('upload_image') 
    return render(request, 'upload_image.html')


