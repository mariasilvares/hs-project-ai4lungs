from django.db import models
from django.contrib.auth.models import User
from django.db.models.signals import post_save
from django.dispatch import receiver
from django.utils import timezone
import os

class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    profile_picture = models.ImageField(upload_to='profile_pictures/', blank=True, null=True)

    def __str__(self):
        return self.user.username

# Criar perfil automaticamente ao criar um usuário
@receiver(post_save, sender=User)
def create_or_save_user_profile(sender, instance, created, **kwargs):
    if created:
        UserProfile.objects.create(user=instance)
    else:
        instance.userprofile.save()

class Activity(models.Model):
    ACTIONS = [
        ('profile_update', 'Profile Change'),
        ('image_upload', 'X-Ray Uploaded'),
    ]
    
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    action = models.CharField(max_length=255, choices=ACTIONS)
    timestamp = models.DateTimeField(auto_now_add=True)
    additional_info = models.TextField(blank=True, null=True)
    details = models.TextField(blank=True, null=True)

    def __str__(self):
        return f"{self.user.username} - {self.get_action_display()}"

class Patient(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='patients')  
    name = models.CharField(max_length=255)  
    number = models.CharField(max_length=50, unique=True)  
    registration_date = models.DateTimeField(auto_now_add=True)  

    def __str__(self):
        return f"{self.name} ({self.number})"

class MedicalImage(models.Model):
    patient = models.ForeignKey(Patient, on_delete=models.CASCADE)
    image = models.ImageField(upload_to='medical_images/')
    description = models.TextField(blank=True, null=True)
    uploaded_at = models.DateTimeField(default=timezone.now)

    def __str__(self):
        return f"Image by {self.patient.name} - {self.uploaded_at.strftime('%Y-%m-%d %H:%M')}"

    # Apaga o arquivo físico ao deletar do banco de dados
    def delete(self, *args, **kwargs):
        if self.image:
            if os.path.exists(self.image.path):
                os.remove(self.image.path)
        super().delete(*args, **kwargs)

class PatientInfo(models.Model):
    patient = models.ForeignKey(Patient, related_name="infos", on_delete=models.CASCADE)
    title = models.CharField(max_length=200)
    description = models.TextField()

    def __str__(self):
        return f"{self.title} - {self.patient.name}"
