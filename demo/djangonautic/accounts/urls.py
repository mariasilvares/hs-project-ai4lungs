from django.urls import path
from . import views

app_name = 'accounts'

urlpatterns = [

    # Autenticação
    path('signup/', views.signup_view, name='signup'),
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),

    # Perfil
    path('profile/', views.profile_view, name='profile'), 
    path('profile/edit/', views.profile_edit, name='profile_edit'), 

    # Pacientes
    path('pacientes/', views.pacientes, name='pacientes'),
    path('paciente/excluir/<int:paciente_id>/', views.excluir_paciente, name='excluir_paciente'),
    path('upload_image/<int:paciente_id>/', views.upload_image, name='upload_image'), 
    path('add_patient_info/<int:paciente_id>/', views.add_patient_info, name='add_patient_info'),
    path('patients/<int:paciente_id>/add_info/', views.add_patient_info, name='add_patient_info'),
    path('delete_patient_info/<int:info_id>/', views.delete_patient_info, name='delete_patient_info'),
    path('delete_patient_info/<int:info_id>/', views.delete_patient_info, name='delete_patient_info'),

    # Processamento de Imagem
    path("run_model/<int:image_id>/", views.run_model, name="run_model"),

]