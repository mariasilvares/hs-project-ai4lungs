from django.urls import re_path,path
from . import views
from django.contrib.auth import views as auth_views
from django.urls import include, path
from django.contrib import admin

app_name = 'accounts'

urlpatterns = [
    re_path(r'^signup/$', views.signup_view, name = 'signup'),
    re_path(r'^login/$', views.login_view, name = 'login'),
    re_path(r'^logout/$', views.logout_view, name = 'logout'),
]

