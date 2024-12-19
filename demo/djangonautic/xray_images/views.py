from django.shortcuts import render, redirect
from .forms import XrayImageForm

def upload_image(request):
    if request.method == 'POST':
        form = XrayImageForm(request.POST, request.FILES)
        if form.is_valid():
            image = form.save(commit=False)
            image.user = request.user
            image.save()
            return redirect('upload_image')
    else:
        form = XrayImageForm()
    return render(request, 'xray_images/upload_image.html', {'form': form})

