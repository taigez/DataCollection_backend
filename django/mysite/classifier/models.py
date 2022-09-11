from django.db import models

# Create your models here.

class Sentences_edu(models.Model):
    body = models.TextField(blank=True)

class Sentences_int(models.Model):
    body = models.TextField(blank=True)

class Sentences_awd(models.Model):
    body = models.TextField(blank=True)

class Sentences_temp_edu(models.Model):
    body = models.TextField(blank=True)

class Sentences_temp_int(models.Model):
    body = models.TextField(blank=True)

class Sentences_temp_awd(models.Model):
    body = models.TextField(blank=True)

class Sentences_irr_edu(models.Model):
    body = models.TextField(blank=True)

class Sentences_irr_awd(models.Model):
    body = models.TextField(blank=True)

class Sentences_irr_int(models.Model):
    body = models.TextField(blank=True)