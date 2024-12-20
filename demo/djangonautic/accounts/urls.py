from django.urls import re_path, path
from . import views
from django.contrib.auth import views as auth_views

app_name = 'accounts'

urlpatterns = [
    re_path(r'^signup/$', views.signup_view, name = 'signup'),
    re_path(r'^login/$', views.login_view, name = 'login'),
    re_path(r'^logout/$', views.logout_view, name = 'logout'),
    path('profile/', views.profile_view, name='profile'),  # Aqui deve ser 'profile_view' ou 'profile_edit'
    path('profile/edit/', views.profile_edit, name='profile_edit'),  # Verifique se a URL está configurada corretamente
]
