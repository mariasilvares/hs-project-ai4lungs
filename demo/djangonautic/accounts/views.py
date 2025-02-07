from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth import login, logout
from .forms import UserEditForm, UserProfileForm,PatientForm, MedicalImageForm
from django.contrib import messages
from .models import Patient, MedicalImage, Activity, PatientInfo
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .inference import predict
from django.shortcuts import render, redirect
from django.contrib import messages
import cv2
from model_utilities import OpenCVXRayNN 

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

            messages.success(request, "Profile updated with success!")
            return redirect('profile')  # Redireciona para o perfil após salvar
        else:
            messages.error(request, "An error occurred when updating the profile. Check the fields and try again.")
    else:
        user_form = UserEditForm(instance=request.user)
        profile_form = UserProfileForm(instance=request.user.userprofile)
    
    return render(request, 'accounts/profile_edit.html', {
        'user_form': user_form,
        'profile_form': profile_form
    })


def upload_image(request, paciente_id):
    # Recupera o paciente com o ID fornecido
    paciente = get_object_or_404(Patient, id=paciente_id)

    if request.method == 'POST' and 'image' in request.FILES:
        uploaded_file = request.FILES['image']
        MedicalImage.objects.create(patient=paciente, image=uploaded_file)

        # Registrar a atividade de upload
        activity = Activity.objects.create(
            user=request.user,
            action=f"Upload of {paciente.name}`s X-Ray",
        )

        # Mensagem de sucesso
        messages.success(request, 'Image uploaded with success!')

    # Obtém todas as imagens para o paciente
    images = MedicalImage.objects.filter(patient=paciente)
    return render(request, 'accounts/medical_image.html', {'paciente': paciente, 'images': images})


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
            messages.success(request, 'Patient added with success!')
            return render(request, 'accounts/pacientes.html', {'form': form, 'pacientes': pacientes})
    else:
        form = PatientForm()

    return render(request, 'accounts/pacientes.html', {'form': form, 'pacientes': pacientes})


def excluir_paciente(request, paciente_id):

    # Obtém o paciente a partir do ID, ou retorna 404 se não encontrado
    paciente = get_object_or_404(Patient, id=paciente_id, user=request.user)

    if request.method == 'POST':
        # Excluir o paciente
        paciente.delete()
        messages.success(request, 'Patient deleted with success!')
        return redirect('accounts:pacientes')  # Redireciona para a lista de pacientes

    return render(request, 'accounts/excluir_paciente.html', {'paciente': paciente})


def medical_image(request, paciente_id):
    paciente = get_object_or_404(Patient, id=paciente_id, user=request.user)
    images = MedicalImage.objects.filter(patient=paciente)
    prediction = None  # Inicializa a variável de predição

    if request.method == 'POST' and 'image' in request.FILES:
        form = MedicalImageForm(request.POST, request.FILES)
        if form.is_valid():
            medical_image = form.save(commit=False)
            medical_image.patient = paciente
            medical_image.save()

            image_path = medical_image.image.path
            prediction = predict(image_path)  # Chama a predição do modelo

            medical_image.description = prediction  # Salva a predição na imagem
            medical_image.save()

            messages.success(request, 'X-ray uploaded successfully!')
            return redirect('accounts:medical_image', paciente_id=paciente.id)

    else:
        form = MedicalImageForm()

    return render(request, 'accounts/medical_image.html', {
        'paciente': paciente,
        'images': images,
        'form': form,
        'prediction': prediction  # Passa a predição para o template
    })

def add_patient_info(request, paciente_id):
    paciente = Patient.objects.get(id=paciente_id)
    
    if request.method == 'POST':
        title = request.POST.get('title')
        description = request.POST.get('description')

        # Cria e salva as informações adicionais
        PatientInfo.objects.create(patient=paciente, title=title, description=description)
        
        # Redireciona para a página de detalhes do paciente
        return redirect('accounts:medical_images', paciente_id=paciente.id)
    
    return render(request, 'accounts/add_patient_info.html', {'paciente': paciente})


@csrf_exempt
def delete_image(request, image_id):
    if request.method == 'DELETE':
        try:
            image = MedicalImage.objects.get(id=image_id)
            image.delete()
            return JsonResponse({'message': 'Image deleted successfully.'}, status=200)
        except MedicalImage.DoesNotExist:
            return JsonResponse({'error': 'Image not found.'}, status=404)
    return JsonResponse({'error': 'Invalid request method.'}, status=405)

@csrf_exempt
def delete_patient_info(request, info_id):
    if request.method == 'DELETE':
        try:
            info = PatientInfo.objects.get(id=info_id)
            info.delete()
            return JsonResponse({'message': 'Information deleted successfully.'}, status=200)
        except PatientInfo.DoesNotExist:
            return JsonResponse({'error': 'Information not found.'}, status=404)
    return JsonResponse({'error': 'Invalid request method.'}, status=405)

def analyze_image(image_path):
    # Carregar e processar a imagem com OpenCV (ou outro pré-processamento necessário)
    image = cv2.imread(image_path)
    processed_image = MedicalImage(image)  # Função para pré-processamento 

    # Carregar o modelo treinado
    model = OpenCVXRayNN()  # Instância do seu modelo
    prediction = model.predict(processed_image)  # Fazendo a predição

    # Interpretar o resultado (assumindo que o modelo retorna um rótulo como 'COVID', 'Pneumonia', 'Normal')
    if prediction == 0:
        return "Normal"
    elif prediction == 1:
        return "COVID"
    elif prediction == 2:
        return "Pneumonia"
    else:
        return "Unknown"