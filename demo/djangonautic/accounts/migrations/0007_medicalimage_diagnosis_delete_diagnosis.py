# Generated by Django 5.1.4 on 2025-02-20 11:36

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('accounts', '0006_alter_medicalimage_description_diagnosis'),
    ]

    operations = [
        migrations.AddField(
            model_name='medicalimage',
            name='diagnosis',
            field=models.CharField(blank=True, max_length=255, null=True),
        ),
        migrations.DeleteModel(
            name='Diagnosis',
        ),
    ]
