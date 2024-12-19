from django.urls import include, re_path, path
from django.contrib import admin
from . import views
from django.contrib.staticfiles.urls import staticfiles_urlpatterns
from django.conf.urls.static import static 
from django.conf import settings
from django.contrib.auth import views as auth_views
from xray_images import views as xray_views


urlpatterns = [
    re_path(r'^admin/', admin.site.urls),
    re_path('xray_images/', include('xray_images.urls')),  # Inclusão do app xray_images
    re_path(r'^accounts/', include('accounts.urls')),
    re_path('articles/', include('articles.urls')), 
    re_path(r'^$', views.homepage, name='home'),  # Home page
    re_path(r'^profile/', views.profile, name='profile'),  # Perfil do usuário
    re_path(r'^upload/', xray_views.upload_image, name='upload_image'),  # Página de upload
    re_path('profile/', views.profile, name='profile'),  # Página do perfil
    re_path('profile/edit/', views.profile_edit, name='profile_edit'),  # Editar perfil
    re_path('password_change/', auth_views.PasswordChangeView.as_view(), name='change_password'),

]

urlpatterns += staticfiles_urlpatterns()
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
