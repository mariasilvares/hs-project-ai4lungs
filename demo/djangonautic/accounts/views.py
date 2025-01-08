from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth import login, logout
from django.contrib.auth.decorators import login_required
from .forms import UserEditForm, UserProfileForm
from .models import Activity
from django.contrib import messages
from django.utils.timezone import now
from .models import Activity
from .models import Patient, MedicalImage
from .forms import PatientForm


# Create your views here.
def signup_view(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            # log the user in
            login(request, user)
            return redirect('home')
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


@login_required
def pacientes(request):
    # Obtém os pacientes do usuário atual
    pacientes = Patient.objects.filter(user=request.user)

    if request.method == 'POST':
        form = PatientForm(request.POST)
        if form.is_valid():
            # Associar o paciente ao usuário logado
            paciente = form.save(commit=False)
            paciente.user = request.user
            paciente.save()
            messages.success(request, 'Paciente adicionado com sucesso!')
            return render(request, 'accounts/pacientes.html', {'form': form, 'pacientes': pacientes})
    else:
        form = PatientForm()

    return render(request, 'accounts/pacientes.html', {'form': form, 'pacientes': pacientes})

@login_required
def excluir_paciente(request, paciente_id):

    # Obtém o paciente a partir do ID, ou retorna 404 se não encontrado
    paciente = get_object_or_404(Patient, id=paciente_id, user=request.user)

    if request.method == 'POST':
        # Excluir o paciente
        paciente.delete()
        messages.success(request, 'Paciente excluído com sucesso!')
        return redirect('accounts:pacientes')  # Redireciona para a lista de pacientes

    return render(request, 'accounts/excluir_paciente.html', {'paciente': paciente})

@login_required
def medical_image(request, paciente_id):
    # Obtém o paciente a partir do ID
    paciente = get_object_or_404(Patient, id=paciente_id, user=request.user)
    
    # Obtém as imagens médicas associadas ao paciente
    images = MedicalImage.objects.filter(patient=paciente)
    
    return render(request, 'accounts/medical_image.html', {'paciente': paciente, 'images': images})

def excluir_paciente(request, paciente_id):
    paciente = get_object_or_404(Patient, id=paciente_id)
    
    if request.method == 'POST':
        paciente.delete()
        messages.success(request, 'Paciente excluído com sucesso!')
        return redirect('accounts:pacientes')
    
    # Se não for um POST, redireciona de volta para a página de pacientes
    return redirect('accounts:pacientes')