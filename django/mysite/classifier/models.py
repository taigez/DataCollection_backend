from django.db import models

# Create your models here.

class Edu_data(models.Model):
    weight = models. IntegerField(blank=True)
    label = models.IntegerField(blank=True)
    text = models.TextField(blank=True)

class Awd_data(models.Model):
    weight = models. IntegerField(blank=True)
    label = models.IntegerField(blank=True)
    text = models.TextField(blank=True)

class Int_data(models.Model):
    weight = models. IntegerField(blank=True)
    label = models.IntegerField(blank=True)
    text = models.TextField(blank=True)
    

class Raw(models.Model):
    body = models.TextField(blank=True)

class RawA(models.Model):
    body = models.TextField(blank=True)

class RawE(models.Model):
    body = models.TextField(blank=True)

class RawI(models.Model):
    body = models.TextField(blank=True)

class Sentences_edu(models.Model):
    body = models.TextField(blank=True)

class Sentences_int(models.Model):
    body = models.TextField(blank=True)

class Sentences_awd(models.Model):
    body = models.TextField(blank=True)

class Sentences_pos(models.Model):
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

class Predicted_total_awd(models.Model):
    body = models.FloatField()

class Predicted_total_edu(models.Model):
    body = models.FloatField()

class Predicted_total_int(models.Model):
    body = models.FloatField()

class Correct_total_awd(models.Model):
    body = models.FloatField()

class Correct_total_edu(models.Model):
    body = models.FloatField()

class Correct_total_int(models.Model):
    body = models.FloatField()

class True_total_awd(models.Model):
    body = models.FloatField()

class True_total_edu(models.Model):
    body = models.FloatField()

class True_total_int(models.Model):
    body = models.FloatField()





