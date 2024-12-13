#Imports
from django.urls import include, re_path, path
from django.contrib import admin
from . import views
from django.contrib.staticfiles.urls import staticfiles_urlpatterns
from django.conf.urls.static import static 
from django.conf import settings
from articles import views as article_views
from xray_images import views as xray_views
from django.views.generic import TemplateView

# urlpatterns = [
#     re_path(r'^admin/', admin.site.urls),
#     re_path(r'^accounts/', include('accounts.urls')),
#     re_path(r'^articles/', include ("articles.urls")),
#     re_path(r'^about/$',views.about),
#     re_path(r'^$', article_views.article_list, name='home'),
# ]

# urlpatterns += staticfiles_urlpatterns()
# urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)


#adicionado
urlpatterns = [
    re_path(r'^admin/', admin.site.urls),
    re_path(r'^accounts/', include('accounts.urls')),
    re_path(r'^articles/', include("articles.urls")),
    re_path(r'^about/$', views.about),
    re_path(r'^$', article_views.article_list, name='home'),
    
    # Adicionando a URL para o upload de imagens de raios-X
    re_path(r'^upload/', xray_views.upload_image, name='upload_image'),  # URL para a view de upload de imagem
]

urlpatterns += staticfiles_urlpatterns()
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)