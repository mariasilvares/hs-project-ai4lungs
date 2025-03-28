from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth import login, logout
from .forms import UserEditForm, UserProfileForm,PatientForm, MedicalImageForm
from django.contrib import messages
from .models import Patient, MedicalImage, Activity, PatientInfo
from django.http import JsonResponse, HttpResponseBadRequest
from django.views.decorators.csrf import csrf_exempt
from accounts.inference import predict_xray
import os

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


def profile_view(request):
    return render(request, 'accounts/profile.html')
    

def profile_edit(request):
    if request.method == 'POST':
        user_form = UserEditForm(request.POST, instance=request.user)
        profile_form = UserProfileForm(request.POST, request.FILES, instance=request.user.userprofile)
        
        if user_form.is_valid() and profile_form.is_valid():
            # Salva o formulário de usuário
            user_form.save()
            
            # Registrar a atividade de alteração de perfil
            Activity.objects.create(
                user=request.user,
                action='profile_update',
                additional_info="Change in user profile."
            )

            messages.success(request, "Profile successfully updated!", extra_tags='profile_updated')
            return redirect('profile')
        else:
            messages.error(request, "An error occurred when updating the profile. Check the fields and try again.")
    else:
        user_form = UserEditForm(instance=request.user)
        profile_form = UserProfileForm(instance=request.user.userprofile)
    
    return render(request, 'accounts/profile_edit.html', {
        'user_form': user_form,
        'profile_form': profile_form
    })


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

             # Registra a atividade
            Activity.objects.create(
                user=request.user,
                action=f"Added Patient {paciente.name}",
                additional_info="A new patient has been registered."
            )

        messages.success(request, 'Patient successfully added!', extra_tags='patient_added')
        return redirect('accounts:pacientes')   
    else:
        form = PatientForm()

    return render(request, 'accounts/pacientes.html', {'form': form, 'pacientes': pacientes})


def excluir_paciente(request, paciente_id):

    # Obtém o paciente a partir do ID, ou retorna 404 se não encontrado
    paciente = get_object_or_404(Patient, id=paciente_id, user=request.user)

    if request.method == 'POST':
        # Excluir o paciente
        paciente.delete()

        # Registra a atividade
        Activity.objects.create(
            user=request.user,
            action=f"Patient {paciente.name} deleted",
            details=f"A patient was deleted"
        )


        messages.success(request, 'Patient successfully removed!', extra_tags='patient_deleted')
        return redirect('accounts:pacientes')   
    return render(request, 'accounts/excluir_paciente.html', {'paciente': paciente})



def add_patient_info(request, paciente_id):
    paciente = Patient.objects.get(id=paciente_id)
    
    title = request.POST.get('title')
    description = request.POST.get('description')

    # Create the new patient info
    patient_info = PatientInfo.objects.create(
        patient=paciente, 
        title=title, 
        description=description
    )
    
    # Check if it's an AJAX request
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        # Return JSON response for AJAX
        return JsonResponse({
            'id': patient_info.id,
            'title': patient_info.title,
            'description': patient_info.description
        })
    
    # Traditional form submission
    messages.success(request, "Information successfully added!", extra_tags='info_added')
    return redirect('accounts:upload_image', paciente_id=paciente.id)

@csrf_exempt
def delete_patient_info(request, info_id):
    if request.method == 'DELETE':
        try:
            # Adicione verificação de permissão
            info = PatientInfo.objects.get(id=info_id)
            if info.patient.user != request.user:
                return JsonResponse({'error': 'Permission denied.'}, status=403)
                
            info.delete()
            return JsonResponse({'message': 'Information deleted successfully.'}, status=200)
        except PatientInfo.DoesNotExist:
            return JsonResponse({'error': 'Information not found.'}, status=404)
    return JsonResponse({'error': 'Invalid request method.'}, status=405)



def upload_image(request, paciente_id):
    import logging
    logger = logging.getLogger('upload_view')
    
    paciente = get_object_or_404(Patient, id=paciente_id)
    
    if request.method == 'POST' and 'image' in request.FILES:
        try:
            uploaded_file = request.FILES['image']
            
            # Cria nova imagem
            new_image = MedicalImage.objects.create(
                patient=paciente, 
                image=uploaded_file,
                description=request.POST.get('description', '')
            )
            logger.info(f"Imagem criada: ID={new_image.id}, Caminho={new_image.image.path}")
            
            # Garante que o arquivo foi salvo antes de prosseguir
            if not os.path.exists(new_image.image.path):
                logger.error(f"File not found after upload: {new_image.image.path}")
                raise FileNotFoundError(f"Image file not found: {new_image.image.path}")
            
            # Executa modelo automaticamente ao fazer upload
            try:
                result = predict_xray(new_image.image.path)
                logger.info(f"Resultado da predição: {result}")
                
                # Verifica se o resultado é válido
                if isinstance(result, int) and result in {0, 1, 2}:
                    label_map = {0: 'covid', 1: 'pneumonia', 2: 'normal'}
                    new_image.diagnosis = label_map[result]
                    logger.info(f"Diagnóstico definido: {new_image.diagnosis}")
                else:
                    logger.warning(f"Resultado inválido do modelo: {result}")
                    new_image.diagnosis = "erro"
            except Exception as e:
                logger.error(f"Erro ao executar predição: {e}", exc_info=True)
                new_image.diagnosis = "erro"
            
            # Salva a imagem com o diagnóstico
            new_image.save()
            
            # Registra atividade
            Activity.objects.create(
                user=request.user,
                action=f"Upload of {paciente.name}'s X-Ray",
            )

            # Se a requisição for AJAX, não adiciona mensagens para evitar que fiquem na sessão
            if request.headers.get('x-requested-with') == 'XMLHttpRequest':
                data = {
                    'id': new_image.id,
                    'image_url': new_image.image.url,
                    'diagnosis': new_image.diagnosis
                }
                return JsonResponse(data)
            else:
                messages.success(request, "Image uploaded successfully!", extra_tags='image_uploaded')
                return redirect('accounts:upload_image', paciente_id=paciente.id)
                
        except Exception as e:
            logger.error(f"Erro no upload: {e}", exc_info=True)
            messages.error(request, f'Error uploading image: {str(e)}')
            
    # Resposta padrão para GET ou erros
    images = MedicalImage.objects.filter(patient=paciente)
    return render(request, 'accounts/medical_image.html', {'paciente': paciente, 'images': images})

def delete_image(request, image_id):
    if request.method == "POST":  # Certifique-se de que está recebendo uma requisição POST
        image = get_object_or_404(MedicalImage, id=image_id)

        # Utiliza o atributo correto 'image' em vez de 'filepath'
        if image.image:
            image_path = image.image.path

            try:
                if os.path.exists(image_path):
                    os.remove(image_path)  # Exclui o arquivo físico
                image.delete()  # Remove do banco de dados

                return JsonResponse({"success": True})
            except Exception as e:
                return JsonResponse({"success": False, "error": str(e)})
        else:
            return JsonResponse({"success": False, "error": "No image file found."})

    return JsonResponse({"success": False, "error": "Invalid request"})


def run_model(request, image_id):
    image = get_object_or_404(MedicalImage, id=image_id)
    image_path = image.image.path  
    result = predict_xray(image_path)  # Retorna 0, 1 ou 2
    
    label_map = {0: 'covid', 1: 'pneumonia', 2: 'normal'}
    result_label = label_map[result]
    
    # Atualiza o diagnóstico da imagem
    image.diagnosis = result_label
    image.save()
    
    return JsonResponse({"prediction": result_label})