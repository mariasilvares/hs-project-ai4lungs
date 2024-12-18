#Imports
from django.shortcuts import render, redirect
from django.contrib.auth.forms import UserChangeForm
from django.contrib.auth.decorators import login_required

def homepage (request):
   # return HttpResponse('homepage')
   return render (request,'homepage.html')


# Perfil do usuário
def profile(request):
    return render(request, 'profile.html')

def profile_edit(request):
    if request.method == 'POST':
        form = UserChangeForm(request.POST, request.FILES, instance=request.user)
        if form.is_valid():
            form.save()
            return redirect('profile')  # Redireciona para a página de perfil após salvar
    else:
        form = UserChangeForm(instance=request.user)
    
    return render(request, 'edit_profile.html', {'form': form})
