from django.urls import re_path, path
from . import views
from django.contrib.auth import views as auth_views

app_name = 'accounts'

urlpatterns = [
    re_path(r'^signup/$', views.signup_view, name = 'signup'),
    re_path(r'^login/$', views.login_view, name = 'login'),
    re_path(r'^logout/$', views.logout_view, name = 'logout'),
    path('profile/', views.profile_view, name='profile'), 
    path('profile/edit/', views.profile_edit, name='profile_edit'), 
    path('pacientes/', views.pacientes, name='pacientes'),
    path('paciente/excluir/<int:paciente_id>/', views.excluir_paciente, name='excluir_paciente'),
    path('paciente/medical_image/<int:paciente_id>/', views.medical_image, name='medical_image'),
    path('upload_image/<int:paciente_id>/', views.upload_image, name='upload_image'), 
    path('add_patient_info/<int:paciente_id>/', views.add_patient_info, name='add_patient_info'),
    path('patients/<int:paciente_id>/', views.medical_image, name='medical_images'),
    path('patients/<int:paciente_id>/add_info/', views.add_patient_info, name='add_patient_info'),
    path('delete_image/<int:image_id>/', views.delete_image, name='delete_image'),
    path('delete_patient_info/<int:info_id>/', views.delete_patient_info, name='delete_patient_info'),
]
