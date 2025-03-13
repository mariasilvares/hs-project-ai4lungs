from django.db import models
from django.contrib.auth.models import User
from django.db.models.signals import post_save
from django.dispatch import receiver
from django.utils import timezone

class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    profile_picture = models.ImageField(upload_to='profile_pictures/', blank=True, null=True)

    def __str__(self):
        return self.user.username

@receiver(post_save, sender=User)
def create_user_profile(sender, instance, created, **kwargs):
    if created:
        UserProfile.objects.create(user=instance)

@receiver(post_save, sender=User)
def save_user_profile(sender, instance, **kwargs):
    instance.userprofile.save()

class Activity(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    action = models.CharField(max_length=255)
    timestamp = models.DateTimeField(auto_now_add=True)
    additional_info = models.TextField(blank=True, null=True)
    details = models.TextField(blank=True, null=True)

    def __str__(self):
        return f"{self.user.username} - {self.action}"

    def get_action_display(self):
        if self.action == 'profile_update':
            return 'Profile Change'
        elif self.action == 'image_upload':
            return 'X-Ray Uploaded'
        return self.action  

class Patient(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='patients')  # Relaciona com User
    name = models.CharField(max_length=255)  # Nome do paciente
    number = models.CharField(max_length=50, unique=True)  # Número único do paciente
    registration_date = models.DateTimeField(auto_now_add=True)  # Data de registro

    def __str__(self):
        return f"{self.name} ({self.number})"

class MedicalImage(models.Model):
    patient = models.ForeignKey(Patient, on_delete=models.CASCADE)
    image = models.ImageField(upload_to='medical_images/')
    description = models.TextField()
    uploaded_at = models.DateTimeField(default=timezone.now)  # Define um valor padrão

    def __str__(self):
        return f"Image by {self.patient.name} - {self.uploaded_at.strftime('%Y-%m-%d %H:%M')}"

class PatientInfo(models.Model):
    patient = models.ForeignKey(Patient, related_name="patientinfo_set", on_delete=models.CASCADE)
    title = models.CharField(max_length=200)
    description = models.TextField()
    # Outros campos relacionados à informação...