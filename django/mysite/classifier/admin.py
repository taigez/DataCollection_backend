from django.contrib import admin

# Register your models here.

from .models import Sentences_awd, Sentences_edu, Sentences_int, Sentences_temp_int, Sentences_temp_awd, Sentences_temp_edu, Sentences_irr_awd, Sentences_irr_edu, Sentences_irr_int

admin.site.register(Sentences_int)
admin.site.register(Sentences_awd)
admin.site.register(Sentences_edu)
admin.site.register(Sentences_temp_int)
admin.site.register(Sentences_temp_awd)
admin.site.register(Sentences_temp_edu)
admin.site.register(Sentences_irr_int)
admin.site.register(Sentences_irr_edu)
admin.site.register(Sentences_irr_awd)